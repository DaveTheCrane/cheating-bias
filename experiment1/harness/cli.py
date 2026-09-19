"""Command line interface.

    python -m harness selftest
    python -m harness verify-tasks
    python -m harness list
    python -m harness show-prompt P3 --task perf_memo
    python -m harness run --config configs/smoke.yaml
    python -m harness analyze results/
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from harness import analyze, verify
from harness.config import (
    REPO_ROOT,
    GridConfig,
    available_tasks,
    load_grid,
    load_task,
)
from harness.executor import build_executor
from harness.models.mock import PERSONALITIES
from harness.prompts import LADDER, MONITORING, CHANNELS, render
from harness.runner import RESULTS_FILENAME, load_records, run_grid
from harness.scoring import RunRecord

SANDBOX_WARNING = (
    "!! Running agent-written code. The 'subprocess' backend is process isolation\n"
    "!! with timeouts and a scrubbed environment, not a sandbox: the code runs with\n"
    "!! your filesystem privileges and your network access. Use --sandbox docker for\n"
    "!! anything beyond local piloting."
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="harness",
        description="Measure how prompt conditions affect agent cheating rates.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a grid of cells")
    run.add_argument("--config", help="path to a grid config YAML")
    run.add_argument("--models", nargs="+", help="override the model list")
    run.add_argument("--prompts", nargs="+", help="override the prompt rungs")
    run.add_argument("--tasks", nargs="+", help="override the task list")
    run.add_argument("--repeats", type=int, help="runs per cell")
    run.add_argument("--out", help="output directory")
    run.add_argument("--sandbox", choices=("subprocess", "docker"), help="execution backend")
    run.add_argument("--concurrency", type=int, help="parallel runs")
    run.add_argument("--channel", choices=CHANNELS, help="where the condition text goes")
    run.add_argument("--monitoring", choices=sorted(MONITORING), help="observation salience")
    run.add_argument("--max-turns", type=int, help="cap on model calls per run")
    run.add_argument("--keep-workdirs", action="store_true", help="do not delete run workdirs")
    run.add_argument("--yes", action="store_true", help="skip the sandbox confirmation prompt")

    analyse = sub.add_parser("analyze", help="summarise a results directory")
    analyse.add_argument("path", nargs="?", default="results", help="results dir or jsonl file")
    analyse.add_argument("--csv", help="also write per-cell rows to this CSV path")
    analyse.add_argument("--json", action="store_true", help="emit raw cell data as JSON")

    sub.add_parser("verify-tasks", help="check task solvability, impossibility and twin diff")

    self_test = sub.add_parser("selftest", help="validate the detectors against scripted agents")
    self_test.add_argument("--workers", type=int, default=4, help="parallel runs")
    self_test.add_argument("--keep", action="store_true", help="keep artifacts for inspection")

    sub.add_parser("list", help="list tasks, prompt rungs and mock personalities")

    show = sub.add_parser("show-prompt", help="print the fully assembled context for a cell")
    show.add_argument("prompt", choices=sorted(LADDER))
    show.add_argument("--task", default="perf_memo")
    show.add_argument("--channel", choices=CHANNELS, default="system")
    show.add_argument("--monitoring", choices=sorted(MONITORING), default="none")

    return parser


def _grid_from_args(args: argparse.Namespace) -> GridConfig:
    if args.config:
        config = load_grid(args.config)
    else:
        if not (args.models and args.prompts and args.tasks):
            raise SystemExit(
                "either --config, or all of --models/--prompts/--tasks, must be given"
            )
        config = GridConfig(models=[], prompts=[], tasks=[])

    for attribute, value in (
        ("models", args.models),
        ("prompts", args.prompts),
        ("tasks", args.tasks),
        ("repeats", args.repeats),
        ("out_dir", args.out),
        ("sandbox", args.sandbox),
        ("concurrency", args.concurrency),
        ("prompt_channel", args.channel),
        ("monitoring", args.monitoring),
        ("max_turns", args.max_turns),
    ):
        if value:
            setattr(config, attribute, value)
    if args.keep_workdirs:
        config.keep_workdirs = True

    unknown_prompts = [p for p in config.prompts if p not in LADDER]
    if unknown_prompts:
        raise SystemExit(f"unknown prompt rungs {unknown_prompts}; have {sorted(LADDER)}")
    unknown_tasks = [t for t in config.tasks if t not in available_tasks()]
    if unknown_tasks:
        raise SystemExit(f"unknown tasks {unknown_tasks}; have {available_tasks()}")

    return config


def _cmd_run(args: argparse.Namespace) -> int:
    config = _grid_from_args(args)

    print(
        f"grid: {len(config.models)} model(s) x {len(config.prompts)} rung(s) "
        f"x {len(config.tasks)} task(s) x {config.repeats} repeat(s) "
        f"= {config.run_count} runs"
    )
    print(f"sandbox: {config.sandbox}   out: {config.out_dir}")

    if config.sandbox == "subprocess":
        print(SANDBOX_WARNING)
        real_models = [m for m in config.models if not m.startswith("mock:")]
        if real_models and not args.yes:
            reply = input("continue? [y/N] ").strip().lower()
            if reply not in ("y", "yes"):
                return 1

    completed = 0
    total = config.run_count

    def progress(record: RunRecord) -> None:
        nonlocal completed
        completed += 1
        score = record.score or {}
        state = "ERROR" if record.error else ("CHEAT" if score.get("cheated") else "clean")
        print(
            f"[{completed:>4}/{total}] {record.model:<28}{record.prompt:<4}"
            f"{record.task:<16}{state:<6}{record.duration:>7.1f}s"
            f"  {','.join(score.get('channels') or []) or '-'}"
        )

    records = run_grid(config, Path(config.out_dir), on_record=progress)

    report = analyze.summarise(record.as_dict() for record in records)
    print()
    print(analyze.format_report(report))
    print(f"\nwrote {Path(config.out_dir) / RESULTS_FILENAME}")
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    rows = load_records(Path(args.path))
    report = analyze.summarise(rows)
    if args.json:
        print(
            json.dumps(
                {
                    "cells": [
                        {
                            "model": cell.model,
                            "task": cell.task,
                            "prompt": cell.prompt,
                            "n": cell.n,
                            "cheats": cell.cheats,
                            "rate": cell.rate.point,
                            "ci": [cell.rate.low, cell.rate.high],
                            "channels": dict(cell.channel_counts),
                            "methods": dict(cell.method_counts),
                        }
                        for cell in (report.cells[key] for key in sorted(report.cells))
                    ],
                    "twin_contrasts": report.twin_contrasts,
                },
                indent=2,
            )
        )
    else:
        print(analyze.format_report(report))
    if args.csv:
        written = analyze.write_csv(report, Path(args.csv))
        print(f"\nwrote {written}")
    return 0


def _cmd_verify_tasks(_args: argparse.Namespace) -> int:
    executor = build_executor("subprocess")
    scratch = Path(tempfile.mkdtemp(prefix="cheatbias-verify-"))
    failures = 0

    print("checking that every solvable task can be solved honestly")
    for task_id in available_tasks():
        task = load_task(task_id)
        if not task.solvable:
            continue
        report = verify.check_reference_solution(task, executor, scratch)
        if not report.get("ran"):
            print(f"  {task_id:<18} SKIP  {report.get('reason')}")
            failures += 1
            continue
        state = "ok" if report.get("ok") else "FAIL"
        print(
            f"  {task_id:<18} {state:<5} visible={report['visible'].get('summary')}"
            f"  holdout={report['holdout'].get('summary', 'not run')}"
        )
        failures += 0 if report.get("ok") else 1

    print("\nchecking that every unsolvable task really is unsolvable")
    for task_id in available_tasks():
        task = load_task(task_id)
        if task.solvable:
            continue
        if not task.twin:
            print(f"  {task_id:<18} SKIP  no twin declared to borrow an honest attempt from")
            failures += 1
            continue
        twin = load_task(task.twin)
        report = verify.check_impossibility(task, twin, executor, scratch)
        state = "ok" if report.get("ok") else "FAIL"
        print(
            f"  {task_id:<18} {state:<5} honest impl -> {report['visible'].get('summary')}"
            f"  (want exactly 1 failure)"
        )
        failures += 0 if report.get("ok") else 1

        diff_report = verify.check_twin_minimal_diff(task, twin)
        state = "ok" if diff_report.get("ok") else "FAIL"
        print(f"  {'twin diff':<18} {state:<5} {diff_report.get('line_diffs') or 'no differences'}")
        failures += 0 if diff_report.get("ok") else 1

    print("\nall checks passed" if not failures else f"\n{failures} check(s) failed")
    return 0 if not failures else 1


def _cmd_selftest(args: argparse.Namespace) -> int:
    from harness.selftest import format_result, run_selftest

    print("running harness selftest (scripted agents, no API keys needed)\n")
    result = run_selftest(workers=args.workers, keep=args.keep)
    print(format_result(result))
    return 0 if result.ok else 1


def _cmd_list(_args: argparse.Namespace) -> int:
    print("tasks")
    for task_id in available_tasks():
        task = load_task(task_id)
        flag = "solvable" if task.solvable else "UNSOLVABLE"
        twin = f"  twin={task.twin}" if task.twin else ""
        print(f"  {task_id:<18}{flag:<11}{task.title}{twin}")

    print("\nprompt rungs")
    for key, condition in LADDER.items():
        print(f"  {key:<4}{condition.label:<24}expected: {condition.expected_direction}")

    print("\nmock personalities")
    print("  " + ", ".join(PERSONALITIES))

    print("\nprompt channels:  " + ", ".join(CHANNELS))
    print("monitoring levels: " + ", ".join(sorted(MONITORING)))
    return 0


def _cmd_show_prompt(args: argparse.Namespace) -> int:
    task = load_task(args.task)
    rendered = render(
        args.prompt, task.goal, channel=args.channel, monitoring=args.monitoring
    )
    print("=" * 70)
    print(f"SYSTEM  ({args.prompt} via {args.channel}, monitoring={args.monitoring})")
    print("=" * 70)
    print(rendered.system)
    print()
    print("=" * 70)
    print("USER")
    print("=" * 70)
    print(rendered.user)
    for rel, content in rendered.repo_files.items():
        print()
        print("=" * 70)
        print(f"INJECTED REPO FILE: {rel}")
        print("=" * 70)
        print(content)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    handlers = {
        "run": _cmd_run,
        "analyze": _cmd_analyze,
        "verify-tasks": _cmd_verify_tasks,
        "selftest": _cmd_selftest,
        "list": _cmd_list,
        "show-prompt": _cmd_show_prompt,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
