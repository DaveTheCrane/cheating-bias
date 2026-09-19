"""Provider-neutral model interface.

The conversation is held in a neutral form and translated by each adapter, so
the agent loop, the tool schemas and the trace format are identical across
providers. That matters here: prompt-condition effects must not be confounded
with differences in how a provider's transcript was assembled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class ToolSchema:
    """A tool advertised to the model."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class ToolCall:
    """A model's request to invoke a tool."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    """The harness's response to a ToolCall."""

    id: str
    content: str
    is_error: bool = False


@dataclass
class Message:
    """One conversation turn.

    An assistant turn carries ``text`` and/or ``tool_calls``. A user turn carries
    ``text`` and/or ``tool_results``.
    """

    role: str
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)


@dataclass
class ModelResponse:
    """One model completion."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


@runtime_checkable
class Model(Protocol):
    """Minimal contract every adapter satisfies."""

    spec: str
    name: str

    def generate(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSchema],
    ) -> ModelResponse: ...


class ModelError(RuntimeError):
    """Raised when a provider call fails in a way the runner should record."""
