"""Provenance stamping, so old results stay interpretable.

Results are only reusable if you can tell what produced them. A row that says
"cheat rate 0.2 for perf_memo" is worthless six weeks later if the task text, the
prompt ladder, or the detector thresholds have moved in the meantime and nothing
recorded which version was in play.

Each run therefore carries a digest of the task (its manifest plus every
workspace file), a digest of the exact rendered prompt, the harness version and
the git commit. Comparing digests across two grids answers "were these measured
against the same thing?" mechanically, instead of by memory.
"""

from __future__ import annotations

import functools
import hashlib
import subprocess
from pathlib import Path
from typing import Any

from harness import __version__
from harness.config import REPO_ROOT, TaskSpec
from harness.workspace import hash_tree


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@functools.lru_cache(maxsize=1)
def git_commit() -> str | None:
    """Short HEAD sha, with a dirty marker when the tree has local changes."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if head.returncode != 0:
            return None
        sha = head.stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return f"{sha}-dirty" if status.stdout.strip() else sha
    except (OSError, subprocess.SubprocessError):
        return None


@functools.lru_cache(maxsize=32)
def task_digest(task_id: str, task_dir: str, workspace_dir: str) -> str:
    """Digest the task manifest and every file the agent will be given."""
    manifest = Path(task_dir) / "task.yaml"
    parts = [manifest.read_text(encoding="utf-8") if manifest.is_file() else ""]
    for rel, digest in sorted(hash_tree(Path(workspace_dir)).items()):
        parts.append(f"{rel}:{digest}")
    return _sha("\n".join(parts))


def prompt_digest(system: str, user: str) -> str:
    """Digest the exact context the model was handed."""
    return _sha(system + "\x00" + user)


def stamp(task: TaskSpec, system: str, user: str, rates_as_of: str) -> dict[str, Any]:
    """Build the provenance block for one run."""
    return {
        "harness_version": __version__,
        "git_commit": git_commit(),
        "task_digest": task_digest(task.id, str(task.task_dir), str(task.workspace_dir)),
        "prompt_digest": prompt_digest(system, user),
        "rates_as_of": rates_as_of,
    }
