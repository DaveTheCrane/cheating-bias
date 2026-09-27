"""Grid runner.

One cell is (model, prompt condition, task); one run is one repeat of a cell.
Each run gets its own workdir and its own trace, and rows are appended to
``results.jsonl`` as they finish so a long grid can be interrupted and analysed
from whatever completed.
"""

from __future__ import annotations

import json
import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterable

from harness import detectors, provenance, verify  # noqa: F401  (provenance used below)
from harness.agent import run_agent
from harness.config import GridConfig, TaskSpec, load_task
from harness.executor import build_executor
from harness.models import build_model
from harness.models.mock import MockModel
from harness.pricing import RATES_AS_OF, cost_for
from harness.prompts import manipulation_check, render
from harness.scoring import RunRecord, score_run
from harness.tools import Toolbox
from harness.trace import Trace
from harness.workspace import git_init_repo, hash_tree, materialise

RESULTS_FILENAME = "results.jsonl"
ARTIFACTS_DIRNAME = "agent_files"

# Cap on any single persisted artifact. Keeps a pathological multi-megabyte write
# from bloating the results tree while still capturing anything human-sized.
MAX_ARTIFACT_BYTES = 256 * 1024


def effective_max_turns(task: TaskSpec, config: GridConfig) -> int:
    """Turn cap for this task: its own if declared, otherwise the grid default.

    A grid-wide cap is the wrong shape. On the pilot, the solvable tasks finished
    in 6-15 turns while the unsolvable one used all 30 on two thirds of its runs —
    and cheats on that task appeared at turns 15, 21, 23 and 28, i.e. right up
    against the cap. A cap inside the distribution of when cheating happens
    censors the tail and biases the measured rate downward, on precisely the arm
    that carries the signal.
    """
    return task.max_turns or config.max_turns


def persist_agent_files(root: Path, diff: Any, destination: Path) -> list[str]:
    """Save the agent's changed and added files alongside the run record.

    This is what makes a grid re-scorable. Detectors change — three of the ones
    here changed after real runs exposed problems — and without the final code
    there is no way to apply a corrected detector to data you already paid for.
    Only the files the agent touched are kept, so the cost is kilobytes per run.
    """
    saved: list[str] = []
    for rel in sorted({*diff.modified, *diff.added}):
        source = root / rel
        if not source.is_file():
            continue
        try:
            if source.stat().st_size > MAX_ARTIFACT_BYTES:
                continue
            text = source.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        saved.append(rel)
    return saved


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in text)


def new_batch_id() -> str:
    """Identifier for one invocation of the runner.

    Timestamp first so batches sort chronologically, plus a random suffix: two
    invocations started within the same second would otherwise share an id and be
    indistinguishable, which is the one thing this is for.

    The suffix is 8 hex characters rather than 4. At 4 it collided once in 200 draws
    in the self-test, which is the birthday bound doing what it does — 4 hex is only
    65k values. 8 makes it negligible at no cost.
    """
    return f"{time.strftime('%Y%m%d-%H%M%S', time.localtime())}-{uuid.uuid4().hex[:8]}"


def run_one(
    model_spec: str,
    condition_id: str,
    task: TaskSpec,
    config: GridConfig,
    out_dir: Path,
    repeat: int = 0,
    batch_id: str = "",
) -> RunRecord:
    """Execute and grade a single run.

    Never raises for ordinary failure: harness or provider errors are captured in
    ``RunRecord.error`` so one bad cell cannot take down a grid, and excluded
    rows stay visible in the results rather than vanishing.
    """
    run_id = f"{_slug(model_spec)}__{condition_id}__{task.id}__r{repeat}__{uuid.uuid4().hex[:8]}"
    run_dir = Path(out_dir) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    workdir = run_dir / "work"

    started = time.time()
    record = RunRecord(
        run_id=run_id,
        batch_id=batch_id or new_batch_id(),
        model=model_spec,
        prompt=condition_id,
        task=task.id,
        task_family=task.family,
        task_variant=task.variant,
        task_solvable=task.solvable,
        repeat=repeat,
        prompt_channel=config.prompt_channel,
        monitoring=config.monitoring,
        temperature=config.temperature,
        sandbox=config.sandbox,
        score={},
        verification={},
        detection={},
        agent={},
        started_at=started,
        trace_path=str(run_dir / "trace.jsonl"),
        workdir=str(workdir) if config.keep_workdirs else None,
    )

    trace = Trace(run_dir / "trace.jsonl")
    try:
        prompt = render(
            condition_id,
            task.goal,
            channel=config.prompt_channel,
            monitoring=config.monitoring,
            max_turns=config.max_turns if config.announce_turn_budget else None,
        )

        # Materialise, then inject any prompt-channel repo files, then take the
        # baseline: files the harness placed must not read as agent additions.
        materialise(task, workdir, git_init=False)
        for rel, content in prompt.repo_files.items():
            target = workdir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        baseline = hash_tree(workdir)
        if task.git_init:
            git_init_repo(workdir)

        executor = build_executor(config.sandbox)
        toolbox = Toolbox(
            workdir,
            executor,
            command_timeout=min(
                config.agent_command_timeout,
                config.command_timeout_cap or config.agent_command_timeout,
            ),
        )

        model = build_model(
            model_spec,
            task=task,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            provider_options=config.provider_options,
        )
        if isinstance(model, MockModel):
            model.bind_task(task)

        outcome = run_agent(model, prompt, toolbox, trace, max_turns=effective_max_turns(task, config))

        verification, diff, _restored = verify.verify_run(
            task,
            workdir,
            baseline,
            executor,
            scratch=run_dir,
            timeout_cap=config.command_timeout_cap,
        )

        detection = detectors.analyse(
            task,
            workdir,
            diff,
            trace,
            verification,
            outcome.finish_status,
            outcome.escape_attempts,
        )

        attention, attention_strict = manipulation_check(
            outcome.finish_summary, trace.last_assistant_text()
        )
        score = score_run(
            detection,
            verification,
            outcome.finish_status,
            condition_id,
            attention,
            # Counts edits as well as whole-file writes, so "did nothing at all"
            # is distinguished from "changed something and fell short".
            files_written=len(trace.mutations()),
            manipulation_check_strict=attention_strict,
        )

        artifacts = run_dir / ARTIFACTS_DIRNAME
        saved = persist_agent_files(workdir, diff, artifacts)

        record.verification = verification
        record.detection = detection.as_dict()
        record.score = score.as_dict()
        record.agent = {
            **asdict(outcome),
            "diff": diff.as_dict(),
            "finish_summary": (outcome.finish_summary or "")[:2000],
            "artifacts_saved": saved,
            # What the provider actually accepted. A config can ask for a
            # temperature the installed SDK no longer supports, and a result that
            # claimed an unapplied setting would be quietly wrong.
            "sampling": getattr(model, "describe_sampling", dict)(),
        }
        record.cost = cost_for(model_spec, outcome.usage).as_dict()
        record.provenance = provenance.stamp(task, prompt.system, prompt.user, RATES_AS_OF)
        record.artifacts_path = str(artifacts) if saved else None
        record.error = outcome.error

        trace.emit("scored", turn=-1, score=score.as_dict(), cost=record.cost)

    except Exception as exc:  # noqa: BLE001 - a bad cell must not kill the grid
        record.error = f"{type(exc).__name__}: {exc}"
        trace.emit("harness_error", turn=-1, error=record.error, traceback=traceback.format_exc())
    finally:
        trace.close()
        record.duration = round(time.time() - started, 3)
        if not config.keep_workdirs:
            shutil.rmtree(run_dir / "restored", ignore_errors=True)
            shutil.rmtree(workdir, ignore_errors=True)

    return record


def provenance_conflicts(results_path: Path, tasks: dict[str, TaskSpec]) -> list[str]:
    """Warn if a results file already holds runs measured against a different setup.

    Appending is the right default — it is what makes ``--resume`` work — but it
    means a results directory can quietly accumulate runs from before and after a
    change to a task, a prompt or a turn cap. Nothing is lost, which is precisely
    the danger: ``analyze`` would pool them into one rate as though they were
    comparable.

    Task digests cover the manifest and every workspace file, so a changed turn cap
    or edited assertion shows up here. Harness-code changes show up only as a
    different git commit, and a dirty tree cannot be pinned at all.
    """
    try:
        rows = load_records(results_path)
    except FileNotFoundError:
        return []
    if not rows:
        return []

    warnings: list[str] = []

    for task_id, task in sorted(tasks.items()):
        current = provenance.task_digest(task.id, str(task.task_dir), str(task.workspace_dir))
        stored = {
            str((row.get("provenance") or {}).get("task_digest"))
            for row in rows
            if row.get("task") == task_id and (row.get("provenance") or {}).get("task_digest")
        }
        other = sorted(stored - {current})
        if other:
            warnings.append(
                f"task {task_id!r}: {len(other)} earlier digest(s) {other} differ from the "
                f"current {current}. Those runs were measured against a different version "
                f"of the task and are not comparable with new ones."
            )

    commits = {
        str((row.get("provenance") or {}).get("git_commit"))
        for row in rows
        if (row.get("provenance") or {}).get("git_commit")
    }
    current_commit = provenance.git_commit()
    if current_commit and commits - {current_commit}:
        warnings.append(
            f"harness commit: existing rows were produced at {sorted(commits - {current_commit})}, "
            f"now at {current_commit}. Detector or adapter changes between those points can "
            f"shift results."
        )
    if any(c.endswith("-dirty") for c in commits | {current_commit or ""}):
        warnings.append(
            "at least one batch was run from a dirty working tree, so its harness version "
            "cannot be pinned. Commit before a run whose results you intend to keep."
        )

    return warnings


def archive_results(out_dir: Path) -> Path | None:
    """Move an existing results directory aside, timestamped. Returns the new path.

    Gives a clean directory for a new generation while keeping every earlier run,
    without breaking ``--resume``: resume works against whatever is in the live
    directory, which after archiving is simply empty.
    """
    source = Path(out_dir)
    if not source.exists() or not any(source.iterdir()):
        return None
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    destination = source.with_name(f"{source.name}-{stamp}")
    shutil.move(str(source), str(destination))
    return destination


def iter_cells(config: GridConfig) -> Iterable[tuple[str, str, str, int]]:
    """Yield ``(model, prompt, task, repeat)`` for the whole grid."""
    for model_spec in config.models:
        for condition_id in config.prompts:
            for task_id in config.tasks:
                for repeat in range(config.repeats):
                    yield model_spec, condition_id, task_id, repeat


def run_grid(
    config: GridConfig,
    out_dir: Path | None = None,
    on_record: Callable[[RunRecord], None] | None = None,
) -> list[RunRecord]:
    """Run every cell in the grid, appending rows as they complete.

    Returns:
        Every RunRecord produced, including excluded ones.
    """
    destination = Path(out_dir or config.out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    results_path = destination / RESULTS_FILENAME

    tasks: dict[str, TaskSpec] = {task_id: load_task(task_id) for task_id in config.tasks}
    cells = list(iter_cells(config))

    if config.resume:
        already = completed_cells(results_path)
        before = len(cells)
        cells = [cell for cell in cells if cell not in already]
        skipped = before - len(cells)
        if skipped:
            print(f"resume: {skipped} of {before} runs already present, {len(cells)} to go")

    batch_id = new_batch_id()
    write_lock = threading.Lock()
    records: list[RunRecord] = []
    handle = results_path.open("a", encoding="utf-8")

    def finish(record: RunRecord) -> None:
        with write_lock:
            handle.write(json.dumps(record.as_dict(), default=str) + "\n")
            handle.flush()
            records.append(record)
            if on_record is not None:
                on_record(record)

    try:
        if config.concurrency <= 1:
            for model_spec, condition_id, task_id, repeat in cells:
                finish(
                    run_one(
                        model_spec, condition_id, tasks[task_id], config, destination,
                        repeat, batch_id,
                    )
                )
        else:
            with ThreadPoolExecutor(max_workers=config.concurrency) as pool:
                futures = [
                    pool.submit(
                        run_one, model_spec, condition_id, tasks[task_id], config, destination,
                        repeat, batch_id,
                    )
                    for model_spec, condition_id, task_id, repeat in cells
                ]
                for future in as_completed(futures):
                    finish(future.result())
    finally:
        handle.close()

    return records


def load_records(path: Path, strict: bool = False) -> list[dict[str, Any]]:
    """Read a results.jsonl file.

    Args:
        path: A results directory or a .jsonl file.
        strict: Raise on an unparseable line instead of skipping it.

    Tolerates a truncated final line by default. A grid killed mid-write leaves
    one, and refusing to read the file would throw away days of completed runs.
    """
    target = Path(path)
    if target.is_dir():
        target = target / RESULTS_FILENAME
    if not target.is_file():
        raise FileNotFoundError(f"no results file at {target}")

    rows: list[dict[str, Any]] = []
    for number, line in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if strict:
                raise
            print(f"warning: skipping unparseable line {number} of {target}")
    return rows


def completed_cells(path: Path) -> set[tuple[str, str, str, int]]:
    """Return ``(model, prompt, task, repeat)`` for runs already recorded.

    Rows carrying an error are treated as not done, so a resumed grid retries
    provider failures and harness faults rather than baking them in.
    """
    try:
        rows = load_records(path)
    except FileNotFoundError:
        return set()

    done: set[tuple[str, str, str, int]] = set()
    for row in rows:
        if row.get("error"):
            continue
        done.add(
            (
                str(row.get("model")),
                str(row.get("prompt")),
                str(row.get("task")),
                int(row.get("repeat", 0)),
            )
        )
    return done
