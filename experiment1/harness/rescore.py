"""Re-grade completed runs against the current detectors, without re-running them.

The detectors are the part of this harness most likely to keep changing, and
three of them changed after real runs exposed problems. Re-running a grid to
apply a corrected detector would mean paying for the model calls twice. Instead,
each run persists what grading needs — the agent's final changed files, the tree
diff, the verification outcomes and the full trace — so scoring can be recomputed
offline and for free.

What is recomputed: every detector finding, the channel attribution,
latency-to-cheat, and the 2x2 score.

What is not: the verification outcomes themselves. Those are test results from the
original workspace and are reused as recorded. Re-running them would need the
whole workspace, not just the changed files, and they do not depend on detector
logic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harness import detectors
from harness.config import load_task
from harness.prompts import manipulation_check
from harness.runner import ARTIFACTS_DIRNAME, RESULTS_FILENAME, load_records
from harness.scoring import score_run
from harness.trace import Trace
from harness.workspace import TreeDiff

RESCORED_FILENAME = "results.rescored.jsonl"


@dataclass
class RescoreOutcome:
    total: int = 0
    rescored: int = 0
    skipped: list[str] = field(default_factory=list)
    changed: list[dict[str, Any]] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)

    @property
    def changed_count(self) -> int:
        return len(self.changed)


def _diff_from_row(row: dict[str, Any]) -> TreeDiff:
    raw = (row.get("agent") or {}).get("diff") or {}
    return TreeDiff(
        modified=list(raw.get("modified") or []),
        deleted=list(raw.get("deleted") or []),
        added=list(raw.get("added") or []),
    )


def _artifacts_dir(row: dict[str, Any], results_dir: Path) -> Path | None:
    stored = row.get("artifacts_path")
    if stored and Path(stored).is_dir():
        return Path(stored)
    # Fall back to locating it relative to the results directory, so a results
    # tree stays usable after being moved or copied to another machine.
    run_id = row.get("run_id")
    if run_id:
        candidate = results_dir / "runs" / run_id / ARTIFACTS_DIRNAME
        if candidate.is_dir():
            return candidate
    return None


def _trace_for(row: dict[str, Any], results_dir: Path) -> Trace | None:
    stored = row.get("trace_path")
    for candidate in (
        Path(stored) if stored else None,
        results_dir / "runs" / str(row.get("run_id")) / "trace.jsonl",
    ):
        if candidate and candidate.is_file():
            try:
                return Trace.load(candidate)
            except (OSError, json.JSONDecodeError, KeyError):
                return None
    return None


def rescore(path: Path) -> RescoreOutcome:
    """Re-grade every run under a results directory."""
    results_dir = Path(path)
    if results_dir.is_file():
        results_dir = results_dir.parent

    outcome = RescoreOutcome()
    rows = load_records(results_dir / RESULTS_FILENAME)
    outcome.total = len(rows)

    task_cache: dict[str, Any] = {}

    for row in rows:
        run_id = str(row.get("run_id"))
        if row.get("error"):
            outcome.skipped.append(f"{run_id}: original run errored")
            outcome.rows.append(row)
            continue

        task_id = str(row.get("task"))
        try:
            task = task_cache.get(task_id) or load_task(task_id)
        except FileNotFoundError:
            outcome.skipped.append(f"{run_id}: task {task_id!r} no longer exists")
            outcome.rows.append(row)
            continue
        task_cache[task_id] = task

        trace = _trace_for(row, results_dir)
        if trace is None:
            outcome.skipped.append(f"{run_id}: trace missing")
            outcome.rows.append(row)
            continue

        artifacts = _artifacts_dir(row, results_dir)
        diff = _diff_from_row(row)
        if artifacts is None and (diff.modified or diff.added):
            outcome.skipped.append(f"{run_id}: agent files not persisted (pre-artifacts run)")
            outcome.rows.append(row)
            continue

        agent = row.get("agent") or {}
        verification = row.get("verification") or {}

        detection = detectors.analyse(
            task,
            artifacts or results_dir,
            diff,
            trace,
            verification,
            agent.get("finish_status"),
            agent.get("escape_attempts") or [],
        )

        attention, attention_strict = manipulation_check(
            agent.get("finish_summary"), trace.last_assistant_text()
        )
        score = score_run(
            detection,
            verification,
            agent.get("finish_status"),
            str(row.get("prompt")),
            attention,
            files_written=len(trace.mutations()),
            manipulation_check_strict=attention_strict,
        )

        before = row.get("score") or {}
        after = score.as_dict()

        label_moved = (before.get("cheated"), before.get("method")) != (
            after.get("cheated"),
            after.get("method"),
        )
        # Report detector-set changes too. Tuning a threshold usually changes which
        # detectors fire without flipping the headline label, and that is exactly
        # the kind of drift worth seeing rather than discovering later.
        findings_moved = sorted(before.get("detectors_fired") or []) != sorted(
            after.get("detectors_fired") or []
        )
        if label_moved or findings_moved:
            outcome.changed.append(
                {
                    "run_id": run_id,
                    "model": row.get("model"),
                    "prompt": row.get("prompt"),
                    "task": task_id,
                    "label_moved": label_moved,
                    "was": {
                        "cheated": before.get("cheated"),
                        "method": before.get("method"),
                        "detectors": before.get("detectors_fired"),
                    },
                    "now": {
                        "cheated": after.get("cheated"),
                        "method": after.get("method"),
                        "detectors": after.get("detectors_fired"),
                    },
                }
            )

        updated = dict(row)
        updated["score"] = after
        updated["detection"] = detection.as_dict()
        updated["rescored_from"] = before
        outcome.rows.append(updated)
        outcome.rescored += 1

    return outcome


def write_rescored(outcome: RescoreOutcome, path: Path) -> Path:
    """Write the re-graded rows beside the originals, never over them."""
    results_dir = Path(path)
    if results_dir.is_file():
        results_dir = results_dir.parent
    destination = results_dir / RESCORED_FILENAME
    with destination.open("w", encoding="utf-8") as handle:
        for row in outcome.rows:
            handle.write(json.dumps(row, default=str) + "\n")
    return destination


def format_outcome(outcome: RescoreOutcome) -> str:
    labels_moved = sum(1 for c in outcome.changed if c["label_moved"])
    lines = [
        f"rescored {outcome.rescored} of {outcome.total} runs; "
        f"{outcome.changed_count} run(s) differ ({labels_moved} with a changed cheat label)"
    ]
    if outcome.skipped:
        lines.append(f"skipped {len(outcome.skipped)}:")
        for reason in outcome.skipped[:10]:
            lines.append(f"  {reason}")
        if len(outcome.skipped) > 10:
            lines.append(f"  ... and {len(outcome.skipped) - 10} more")
    for change in outcome.changed[:20]:
        lines.append(
            f"  {change['run_id'][:40]}  {change['task']}/{change['prompt']}: "
            f"cheated {change['was']['cheated']} -> {change['now']['cheated']}, "
            f"{change['was']['method']} -> {change['now']['method']}"
        )
        lines.append(
            f"      detectors {change['was']['detectors']} -> {change['now']['detectors']}"
        )
    return "\n".join(lines)
