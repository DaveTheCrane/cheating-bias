"""Command execution backends.

SECURITY: running an agent's code means executing untrusted code. The
``subprocess`` backend is *not* a sandbox — it is process isolation with
timeouts, a scrubbed environment and a couple of rlimits, nothing more. Code
written by the agent runs with the harness user's filesystem privileges. Use the
``docker`` backend for anything beyond local piloting: it adds a container
boundary and ``--network none``, which also closes the answer-lookup channel and
prevents exfiltration.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Environment variables that may reach agent-executed code. Everything else is
# dropped, so provider API keys in the harness environment are not readable by
# code the agent writes.
ENV_ALLOWLIST = ("PATH", "LANG", "LC_ALL", "TZ", "HOME", "TMPDIR", "SYSTEMROOT")

MAX_CAPTURE_CHARS = 40_000
DEFAULT_IMAGE = "python:3.12-slim"


@dataclass
class CommandResult:
    """Outcome of one command execution."""

    command: str
    returncode: int
    stdout: str
    stderr: str
    duration: float
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    @property
    def output(self) -> str:
        parts = []
        if self.stdout.strip():
            parts.append(self.stdout.rstrip())
        if self.stderr.strip():
            parts.append("[stderr]\n" + self.stderr.rstrip())
        if self.timed_out:
            parts.append(f"[timed out after {self.duration:.1f}s]")
        return "\n".join(parts) or "(no output)"

    def truncated_output(self, limit: int = 6000) -> str:
        text = self.output
        if len(text) <= limit:
            return text
        head = text[: limit // 2]
        tail = text[-limit // 2 :]
        omitted = len(text) - len(head) - len(tail)
        return f"{head}\n... [{omitted} characters omitted] ...\n{tail}"


def _truncate(text: str) -> str:
    if len(text) <= MAX_CAPTURE_CHARS:
        return text
    return text[:MAX_CAPTURE_CHARS] + f"\n... [truncated at {MAX_CAPTURE_CHARS} chars]"


def _scrubbed_env(workdir: Path, extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {key: os.environ[key] for key in ENV_ALLOWLIST if key in os.environ}
    env["PYTHONPATH"] = str(workdir)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    # Keep pytest's own behaviour deterministic and non-interactive.
    env["PY_COLORS"] = "0"
    env["NO_COLOR"] = "1"
    if extra:
        env.update(extra)
    return env


class SubprocessExecutor:
    """Runs commands directly on the host. Fast, weakly isolated."""

    name = "subprocess"

    def __init__(self, python: str | None = None, cpu_grace: int = 30) -> None:
        self.python = python or sys.executable
        self.cpu_grace = cpu_grace

    def _resolve(self, argv: list[str]) -> list[str]:
        # Tasks declare "python"; bind it to the interpreter running the harness
        # so verification does not depend on what is first on PATH.
        if argv and argv[0] in ("python", "python3"):
            return [self.python, *argv[1:]]
        return list(argv)

    def _limits(self, timeout: int):
        if os.name != "posix":
            return None

        def apply() -> None:  # pragma: no cover - runs in the child process
            import resource

            os.setsid()
            cpu = timeout + self.cpu_grace
            for res, limit in (
                (resource.RLIMIT_CPU, (cpu, cpu + 5)),
                (resource.RLIMIT_FSIZE, (256 * 1024 * 1024,) * 2),
                (resource.RLIMIT_CORE, (0, 0)),
            ):
                try:
                    resource.setrlimit(res, limit)
                except (ValueError, OSError):
                    pass

        return apply

    def run(
        self,
        argv: list[str] | str,
        cwd: Path,
        timeout: int = 90,
        env: dict[str, str] | None = None,
    ) -> CommandResult:
        shell = isinstance(argv, str)
        command = argv if shell else " ".join(self._resolve(argv))
        popen_args: list[str] | str = argv if shell else self._resolve(argv)

        start = time.perf_counter()
        try:
            process = subprocess.Popen(
                popen_args,
                cwd=str(cwd),
                shell=shell,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                text=True,
                errors="replace",
                env=_scrubbed_env(cwd, env),
                preexec_fn=self._limits(timeout),
            )
        except OSError as exc:
            return CommandResult(
                command=command,
                returncode=127,
                stdout="",
                stderr=f"failed to execute: {exc}",
                duration=time.perf_counter() - start,
            )

        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            self._kill(process)
            stdout, stderr = process.communicate()

        return CommandResult(
            command=command,
            returncode=process.returncode if not timed_out else 124,
            stdout=_truncate(stdout or ""),
            stderr=_truncate(stderr or ""),
            duration=time.perf_counter() - start,
            timed_out=timed_out,
        )

    @staticmethod
    def _kill(process: subprocess.Popen) -> None:
        """Kill the whole process group so orphaned children cannot linger."""
        try:
            if os.name == "posix":
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            else:  # pragma: no cover
                process.kill()
        except (ProcessLookupError, PermissionError, OSError):
            try:
                process.kill()
            except OSError:
                pass


class DockerExecutor:
    """Runs commands in a throwaway container with networking disabled."""

    name = "docker"

    def __init__(self, image: str = DEFAULT_IMAGE, memory: str = "1g", cpus: str = "1.0") -> None:
        if shutil.which("docker") is None:
            raise RuntimeError(
                "docker backend requested but the docker CLI is not on PATH. "
                "Use --sandbox subprocess for local piloting."
            )
        self.image = image
        self.memory = memory
        self.cpus = cpus

    def run(
        self,
        argv: list[str] | str,
        cwd: Path,
        timeout: int = 90,
        env: dict[str, str] | None = None,
    ) -> CommandResult:
        if isinstance(argv, str):
            inner = ["/bin/sh", "-lc", argv]
            command = argv
        else:
            inner = list(argv)
            command = " ".join(inner)

        docker_argv = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--memory",
            self.memory,
            "--cpus",
            self.cpus,
            "--pids-limit",
            "256",
            "--workdir",
            "/work",
            "--volume",
            f"{cwd}:/work",
            "--env",
            "PYTHONPATH=/work",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
        ]
        for key, value in (env or {}).items():
            docker_argv += ["--env", f"{key}={value}"]
        docker_argv += [self.image, *inner]

        start = time.perf_counter()
        try:
            completed = subprocess.run(
                docker_argv,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout + 30,
            )
        except subprocess.TimeoutExpired:
            return CommandResult(
                command=command,
                returncode=124,
                stdout="",
                stderr="docker run exceeded the harness timeout",
                duration=time.perf_counter() - start,
                timed_out=True,
            )

        return CommandResult(
            command=command,
            returncode=completed.returncode,
            stdout=_truncate(completed.stdout or ""),
            stderr=_truncate(completed.stderr or ""),
            duration=time.perf_counter() - start,
        )


def build_executor(kind: str = "subprocess", **kwargs: object):
    """Return an executor by name."""
    if kind == "subprocess":
        return SubprocessExecutor(**kwargs)  # type: ignore[arg-type]
    if kind == "docker":
        return DockerExecutor(**kwargs)  # type: ignore[arg-type]
    raise ValueError(f"unknown sandbox backend {kind!r}; expected 'subprocess' or 'docker'")
