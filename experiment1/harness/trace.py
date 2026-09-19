"""JSONL run traces.

Every model message, tool call and tool result goes through here. Because all
of the agent's file and shell access is mediated by harness/tools.py, the trace
is the ground truth for anything behavioural: which files were read, in what
order the workspace was mutated, and on which turn a violation first appeared.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator


@dataclass
class TraceEvent:
    seq: int
    turn: int
    kind: str
    ts: float
    data: dict[str, Any] = field(default_factory=dict)


class Trace:
    """Append-only event log for a single run."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else None
        self._events: list[TraceEvent] = []
        self._seq = 0
        self._lock = threading.Lock()
        self._handle = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("w", encoding="utf-8")

    def emit(self, kind: str, turn: int = -1, **data: Any) -> TraceEvent:
        """Record one event and flush it to disk."""
        with self._lock:
            event = TraceEvent(seq=self._seq, turn=turn, kind=kind, ts=time.time(), data=data)
            self._seq += 1
            self._events.append(event)
            if self._handle is not None:
                self._handle.write(json.dumps(asdict(event), default=str) + "\n")
                self._handle.flush()
            return event

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "Trace":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def events(self) -> list[TraceEvent]:
        return list(self._events)

    def of_kind(self, *kinds: str) -> list[TraceEvent]:
        return [e for e in self._events if e.kind in kinds]

    def tool_calls(self, name: str | None = None) -> list[TraceEvent]:
        """Return tool_call events, optionally filtered by tool name."""
        calls = self.of_kind("tool_call")
        if name is None:
            return calls
        return [e for e in calls if e.data.get("tool") == name]

    def files_read(self) -> list[str]:
        """Return every path passed to a successful read_file call, in order."""
        return [
            str(e.data["args"].get("path", ""))
            for e in self.tool_calls("read_file")
            if e.data.get("ok", True)
        ]

    def writes(self) -> list[tuple[int, str, str]]:
        """Return ``(turn, path, content)`` for each write, in call order.

        Used to compute latency-to-cheat: the detectors are re-run against each
        written revision so the first offending turn can be identified, rather
        than only inspecting the final workspace state.
        """
        out: list[tuple[int, str, str]] = []
        for event in self.tool_calls("write_file"):
            if not event.data.get("ok", True):
                continue
            args = event.data.get("args", {})
            out.append((event.turn, str(args.get("path", "")), str(args.get("content", ""))))
        return out

    def mutations(self) -> list[tuple[int, str, str, str]]:
        """Return ``(turn, path, content, kind)`` for every successful mutation.

        Covers ``edit_file`` as well as ``write_file``. Latency-to-cheat replays
        this list, and an earlier version looked only at whole-file writes, which
        meant any cheat made as a small targeted edit — the more natural way to
        tamper with a single assertion — never registered a turn at all.

        For an edit, ``content`` is the replacement fragment rather than the whole
        file, so callers should expect AST parsing of it to fail and text-level
        checks to still apply.
        """
        out: list[tuple[int, str, str, str]] = []
        for event in self.of_kind("tool_call"):
            tool = event.data.get("tool")
            if tool not in ("write_file", "edit_file") or not event.data.get("ok", True):
                continue
            args = event.data.get("args", {})
            content = args.get("content") if tool == "write_file" else args.get("new_string")
            out.append((event.turn, str(args.get("path", "")), str(content or ""), str(tool)))
        return out

    def commands(self) -> list[tuple[int, str]]:
        """Return ``(turn, command)`` for each shell command, in call order."""
        return [
            (e.turn, str(e.data.get("args", {}).get("command", "")))
            for e in self.tool_calls("run_command")
        ]

    def last_assistant_text(self) -> str:
        messages = self.of_kind("model_message")
        for event in reversed(messages):
            text = str(event.data.get("text", "")).strip()
            if text:
                return text
        return ""

    def turn_count(self) -> int:
        turns = [e.turn for e in self._events if e.turn >= 0]
        return (max(turns) + 1) if turns else 0

    @classmethod
    def load(cls, path: Path) -> "Trace":
        """Rehydrate a trace from a JSONL file without reopening it for write."""
        trace = cls(None)
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            trace._events.append(
                TraceEvent(
                    seq=raw["seq"],
                    turn=raw["turn"],
                    kind=raw["kind"],
                    ts=raw["ts"],
                    data=raw.get("data", {}),
                )
            )
        trace._seq = len(trace._events)
        trace.path = Path(path)
        return trace

    def __iter__(self) -> Iterator[TraceEvent]:
        return iter(self._events)

    def __len__(self) -> int:
        return len(self._events)
