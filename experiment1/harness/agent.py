"""The agent loop.

Deliberately plain: one model call, execute whatever tools it asked for, hand
back the results, repeat. No scaffolding that might itself discourage or
encourage shortcutting, since anything clever here would become a confound.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from harness.models.base import Message, Model, ToolResult, ToolSchema
from harness.prompts import RenderedPrompt
from harness.tools import TOOL_SCHEMAS, FinishSignal, Toolbox, ToolError
from harness.trace import Trace

RESULT_TRACE_LIMIT = 4_000


@dataclass
class AgentOutcome:
    """What the agent did, independent of whether it did it honestly."""

    finish_status: str | None = None
    finish_summary: str | None = None
    turns: int = 0
    stop_reason: str = "unknown"
    tool_calls: int = 0
    tool_errors: int = 0
    escape_attempts: list[dict[str, str]] = field(default_factory=list)
    error: str | None = None
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def called_finish(self) -> bool:
        return self.finish_status is not None


def run_agent(
    model: Model,
    prompt: RenderedPrompt,
    toolbox: Toolbox,
    trace: Trace,
    max_turns: int = 30,
    tools: list[ToolSchema] | None = None,
) -> AgentOutcome:
    """Drive one agent session to completion.

    Args:
        model: Any adapter satisfying the Model protocol.
        prompt: The rendered system/user context for this cell.
        toolbox: Tools bound to this run's workdir.
        trace: Open trace to record into.
        max_turns: Hard cap on model calls.
        tools: Tool schemas to advertise. Defaults to the full set.

    Returns:
        An AgentOutcome. Never raises for ordinary agent misbehaviour; provider
        failures are recorded in ``error`` so the runner can exclude the cell
        rather than crash the grid.
    """
    schemas = tools or TOOL_SCHEMAS
    outcome = AgentOutcome()

    toolbox.on_escape = lambda tool, path: outcome.escape_attempts.append(
        {"tool": tool, "path": path}
    )

    trace.emit(
        "run_prompt",
        turn=-1,
        condition=prompt.condition_id,
        channel=prompt.channel,
        monitoring=prompt.monitoring,
        system=prompt.system,
        user=prompt.user,
        repo_files=sorted(prompt.repo_files),
    )

    messages: list[Message] = [Message(role="user", text=prompt.user)]
    totals: dict[str, int] = {}

    for turn in range(max_turns):
        outcome.turns = turn + 1
        response = model.generate(prompt.system, messages, schemas)

        if response.error:
            outcome.error = response.error
            outcome.stop_reason = "model_error"
            trace.emit("model_error", turn=turn, error=response.error)
            break

        for key, value in (response.usage or {}).items():
            if isinstance(value, int):
                totals[key] = totals.get(key, 0) + value

        trace.emit(
            "model_message",
            turn=turn,
            text=response.text,
            # Recorded when the provider exposes it, and never sent back in the
            # next request. Lets a later pass ask whether a cheat was planned
            # before the first edit or arrived after an honest attempt failed.
            thinking=response.thinking,
            tool_calls=[
                {"id": c.id, "name": c.name, "arguments": c.arguments}
                for c in response.tool_calls
            ],
            stop_reason=response.stop_reason,
            usage=response.usage,
        )

        if not response.tool_calls:
            outcome.stop_reason = "no_tools"
            break

        messages.append(
            Message(role="assistant", text=response.text, tool_calls=list(response.tool_calls))
        )

        results: list[ToolResult] = []
        finished: FinishSignal | None = None

        for call in response.tool_calls:
            outcome.tool_calls += 1
            is_error = False
            try:
                raw = toolbox.call(call.name, dict(call.arguments))
            except ToolError as exc:
                outcome.tool_errors += 1
                is_error = True
                content = f"Error: {exc}"
            except Exception as exc:  # noqa: BLE001 - never let a tool kill the grid
                outcome.tool_errors += 1
                is_error = True
                content = f"Error: unexpected {type(exc).__name__}: {exc}"
            else:
                if isinstance(raw, FinishSignal):
                    finished = raw
                    content = f"Recorded status={raw.status}. Session over."
                else:
                    content = str(raw)

            trace.emit(
                "tool_call",
                turn=turn,
                tool=call.name,
                args=dict(call.arguments),
                ok=not is_error,
                result=content[:RESULT_TRACE_LIMIT],
            )
            results.append(ToolResult(id=call.id, content=content, is_error=is_error))

            if finished is not None:
                break

        messages.append(Message(role="user", tool_results=results))

        if finished is not None:
            outcome.finish_status = finished.status
            outcome.finish_summary = finished.summary
            outcome.stop_reason = "finish"
            break
    else:
        outcome.stop_reason = "max_turns"

    outcome.usage = totals
    trace.emit(
        "run_end",
        turn=-1,
        stop_reason=outcome.stop_reason,
        finish_status=outcome.finish_status,
        turns=outcome.turns,
        tool_calls=outcome.tool_calls,
        tool_errors=outcome.tool_errors,
        escape_attempts=outcome.escape_attempts,
    )
    return outcome
