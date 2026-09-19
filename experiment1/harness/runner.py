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

from harness import detectors, verify
from harness.agent import run_agent
from harness.config import GridConfig, TaskSpec, load_task
from harness.executor import build_executor
from harness.models import build_model
from harness.models.mock import MockModel
from harness.prompts import manipulation_check_passed, render
from harness.scoring import RunRecord, score_run
from harness.tools import Toolbox
from harness.trace import Trace
from harness.workspace import git_init_repo, hash_tree, materialise

RESULTS_FILENAME = "results.jsonl"


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in text)


def run_one(
    model_spec: str,
    condition_id: str,
    task: TaskSpec,
    config: GridConfig,
    out_dir: Path,
    repeat: int = 0,
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
        )
        if isinstance(model, MockModel):
            model.bind_task(task)

        outcome = run_agent(model, prompt, toolbox, trace, max_turns=config.max_turns)

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

        score = score_run(
            detection,
            verification,
            outcome.finish_status,
            condition_id,
            manipulation_check_passed(outcome.finish_summary, trace.last_assistant_text()),
            # Counts edits as well as whole-file writes, so "did nothing at all"
            # is distinguished from "changed something and fell short".
            files_written=len(trace.mutations()),
        )

        record.verification = verification
        record.detection = detection.as_dict()
        record.score = score.as_dict()
        record.agent = {
            **asdict(outcome),
            "diff": diff.as_dict(),
            "finish_summary": (outcome.finish_summary or "")[:2000],
        }
        record.error = outcome.error

        trace.emit("scored", turn=-1, score=score.as_dict())

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
                finish(run_one(model_spec, condition_id, tasks[task_id], config, destination, repeat))
        else:
            with ThreadPoolExecutor(max_workers=config.concurrency) as pool:
                futures = [
                    pool.submit(
                        run_one, model_spec, condition_id, tasks[task_id], config, destination, repeat
                    )
                    for model_spec, condition_id, task_id, repeat in cells
                ]
                for future in as_completed(futures):
                    finish(future.result())
    finally:
        handle.close()

    return records


def load_records(path: Path) -> list[dict[str, Any]]:
    """Read a results.jsonl file."""
    target = Path(path)
    if target.is_dir():
        target = target / RESULTS_FILENAME
    if not target.is_file():
        raise FileNotFoundError(f"no results file at {target}")
    rows: list[dict[str, Any]] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows
