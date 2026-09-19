"""Task and grid configuration loading."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
TASKS_ROOT = REPO_ROOT / "tasks"
CONFIGS_ROOT = REPO_ROOT / "configs"


@dataclass(frozen=True)
class CommandSpec:
    """A command the harness runs during verification."""

    argv: list[str]
    timeout: int = 90


@dataclass(frozen=True)
class TaskSpec:
    """Everything the harness needs to run and grade one task."""

    id: str
    family: str
    variant: str
    title: str
    solvable: bool
    goal: str
    task_dir: Path
    workspace_dir: Path
    holdout_dir: Path | None
    protected_paths: list[str] = field(default_factory=list)
    restore_excludes: list[str] = field(default_factory=list)
    commands: dict[str, CommandSpec] = field(default_factory=dict)
    reference: dict[str, str] = field(default_factory=dict)
    tripwire_paths: list[str] = field(default_factory=list)
    detectors: dict[str, dict[str, Any]] = field(default_factory=dict)
    git_init: bool = True
    twin: str | None = None
    impossibility_notes: str | None = None

    def detector_config(self, name: str) -> dict[str, Any]:
        """Return the config block for one detector, defaulting to enabled."""
        return self.detectors.get(name, {"enabled": True})

    def detector_enabled(self, name: str) -> bool:
        return bool(self.detector_config(name).get("enabled", True))


@dataclass
class GridConfig:
    """A grid of experimental cells to run."""

    models: list[str]
    prompts: list[str]
    tasks: list[str]
    repeats: int = 1
    max_turns: int = 30
    sandbox: str = "subprocess"
    temperature: float = 1.0
    prompt_channel: str = "system"
    monitoring: str = "none"
    out_dir: str = "results"
    seed: int = 0
    max_tokens: int = 4096
    keep_workdirs: bool = False
    concurrency: int = 1
    # Ceiling applied on top of each task's own command timeouts. Useful for
    # smoke runs, where a still-naive implementation would otherwise spend the
    # task's full holdout budget grinding through expensive held-out inputs.
    command_timeout_cap: int | None = None
    agent_command_timeout: int = 90

    @property
    def cell_count(self) -> int:
        return len(self.models) * len(self.prompts) * len(self.tasks)

    @property
    def run_count(self) -> int:
        return self.cell_count * self.repeats


def load_task(task_id: str, tasks_root: Path | None = None) -> TaskSpec:
    """Load a TaskSpec from ``tasks/<task_id>/task.yaml``."""
    root = tasks_root or TASKS_ROOT
    task_dir = (root / task_id).resolve()
    manifest = task_dir / "task.yaml"
    if not manifest.is_file():
        raise FileNotFoundError(f"no task.yaml for task {task_id!r} at {manifest}")

    raw = yaml.safe_load(manifest.read_text()) or {}

    workspace_dir = task_dir / raw.get("workspace_dir", "workspace")
    if not workspace_dir.is_dir():
        raise FileNotFoundError(f"task {task_id!r} has no workspace at {workspace_dir}")

    holdout_raw = raw.get("holdout_dir", "holdout")
    holdout_dir: Path | None = (task_dir / holdout_raw).resolve()
    if holdout_dir is not None and not holdout_dir.is_dir():
        holdout_dir = None

    commands = {
        name: CommandSpec(
            argv=list(spec["argv"]),
            timeout=int(spec.get("timeout", 90)),
        )
        for name, spec in (raw.get("commands") or {}).items()
    }

    return TaskSpec(
        id=raw.get("id", task_id),
        family=raw.get("family", task_id),
        variant=raw.get("variant", "default"),
        title=raw.get("title", task_id),
        solvable=bool(raw.get("solvable", True)),
        goal=str(raw.get("goal", "")).strip(),
        task_dir=task_dir,
        workspace_dir=workspace_dir,
        holdout_dir=holdout_dir,
        protected_paths=list(raw.get("protected_paths") or []),
        restore_excludes=list(raw.get("restore_excludes") or []),
        commands=commands,
        reference=dict(raw.get("reference") or {}),
        tripwire_paths=list(raw.get("tripwire_paths") or []),
        detectors=dict(raw.get("detectors") or {}),
        git_init=bool(raw.get("git_init", True)),
        twin=raw.get("twin"),
        impossibility_notes=raw.get("impossibility_notes"),
    )


def available_tasks(tasks_root: Path | None = None) -> list[str]:
    """Return the ids of every task with a task.yaml, sorted."""
    root = tasks_root or TASKS_ROOT
    if not root.is_dir():
        return []
    return sorted(
        entry.name
        for entry in root.iterdir()
        if entry.is_dir() and (entry / "task.yaml").is_file()
    )


def load_grid(path: str | Path) -> GridConfig:
    """Load a GridConfig from a YAML file."""
    config_path = Path(path)
    if not config_path.is_file():
        candidate = CONFIGS_ROOT / config_path.name
        if candidate.is_file():
            config_path = candidate
        else:
            raise FileNotFoundError(f"no grid config at {path}")

    raw = yaml.safe_load(config_path.read_text()) or {}
    known = {f.name for f in dataclasses.fields(GridConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(
            f"{config_path}: unknown grid config keys {sorted(unknown)}; "
            f"known keys are {sorted(known)}"
        )

    missing = [k for k in ("models", "prompts", "tasks") if k not in raw]
    if missing:
        raise ValueError(f"{config_path}: missing required keys {missing}")

    return GridConfig(**raw)
