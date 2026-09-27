"""Anthropic Messages API adapter.

Spec format: ``anthropic:<model-id>``, e.g. ``anthropic:claude-sonnet-4-6``.

Requires the ``anthropic`` package and ``ANTHROPIC_API_KEY``. The key is read by
the SDK from the harness environment and is never passed into the executor
environment, so code the agent writes cannot read it (see executor.ENV_ALLOWLIST).
"""

from __future__ import annotations

import time
from typing import Any

from harness.models.base import (
    Message,
    ModelError,
    ModelResponse,
    ToolCall,
    ToolSchema,
)
from harness.models.params import (
    declared_parameters,
    describe,
    filter_to_signature,
    unsupported_param,
)

MAX_ATTEMPTS = 5
BACKOFF_BASE = 2.0


class AnthropicModel:
    """Thin adapter over ``client.messages.create``."""

    def __init__(
        self,
        model: str,
        temperature: float = 1.0,
        max_tokens: int = 4096,
        effort: str | None = None,
        cache: bool = True,
        cache_ttl: str = "5m",
        client: Any | None = None,
    ) -> None:
        """
        Args:
            model: Anthropic model id.
            temperature: Requested sampling temperature. Silently dropped if the
                installed SDK does not accept it — 1.8.0 removed it outright — and
                the run record reports what was actually applied.
            max_tokens: Output cap.
            effort: Optional ``output_config.effort``, one of low/medium/high/
                xhigh/max. This is the sampling control that replaced temperature.
                Left unset by default so the account default applies.
            cache: Request prompt caching. On by default because an agent loop
                re-sends its whole transcript every turn: measured over a real
                pilot, 91.6% of this adapter's input tokens were repeated prefix,
                and caching cuts input cost by about 80%. Caching does not change
                model output, only billing. The one case where it loses is a
                single-turn run, which pays the 1.25x cache-write premium with
                nothing to reuse — irrelevant here, where runs are 6 to 30 turns.
            cache_ttl: ``5m`` or ``1h``. 5m is right for an agent loop, whose turns
                are seconds apart; 1h doubles the write premium to buy persistence
                nothing here needs.
            client: Injection point for tests.
        """
        self.model = model
        self.spec = f"anthropic:{model}"
        self.name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.effort = effort
        self.cache = cache
        self.cache_ttl = cache_ttl
        self._dropped: list[str] = []
        self._sampling: dict[str, Any] = {}
        self._cache_strategy = "off"
        if client is not None:
            self._client = client
        else:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover
                raise ModelError(
                    "the anthropic package is required for anthropic:* specs "
                    "(pip install anthropic)"
                ) from exc
            self._client = anthropic.Anthropic()

    def describe_sampling(self) -> dict[str, Any]:
        """What the provider actually accepted. Recorded with each run."""
        return {**self._sampling, "cache_strategy": self._cache_strategy}

    # -- translation -------------------------------------------------------

    @staticmethod
    def _tools(tools: list[ToolSchema]) -> list[dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            for tool in tools
        ]

    @staticmethod
    def _messages(messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for message in messages:
            content: list[dict[str, Any]] = []
            if message.role == "assistant":
                if message.text:
                    content.append({"type": "text", "text": message.text})
                for call in message.tool_calls:
                    content.append(
                        {
                            "type": "tool_use",
                            "id": call.id,
                            "name": call.name,
                            "input": call.arguments,
                        }
                    )
            else:
                for result in message.tool_results:
                    block: dict[str, Any] = {
                        "type": "tool_result",
                        "tool_use_id": result.id,
                        "content": result.content,
                    }
                    if result.is_error:
                        block["is_error"] = True
                    content.append(block)
                if message.text:
                    content.append({"type": "text", "text": message.text})
            if content:
                out.append({"role": message.role, "content": content})
        return out

    # -- generation --------------------------------------------------------

    def _cache_param(self) -> dict[str, str]:
        return {"type": "ephemeral", "ttl": self.cache_ttl}

    def _mark_rolling_breakpoint(self, messages: list[dict[str, Any]]) -> bool:
        """Put a cache breakpoint on the final content block.

        A breakpoint caches everything up to *and including* the block it sits on,
        so one marker at the very end of the conversation covers the system prompt,
        the tool schemas and every prior turn. That is why no separate breakpoint is
        needed for the static parts, and why a single rolling marker is enough.

        Returns:
            True if a breakpoint was placed.
        """
        for message in reversed(messages):
            content = message.get("content")
            if isinstance(content, list) and content and isinstance(content[-1], dict):
                content[-1]["cache_control"] = self._cache_param()
                return True
        return False

    def _payload(
        self, system: str, messages: list[Message], tools: list[ToolSchema]
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system": system,
            "messages": self._messages(messages),
        }
        if tools:
            payload["tools"] = self._tools(tools)
        if self.effort:
            payload["output_config"] = {"effort": self.effort}

        if self.cache:
            # Prefer the top-level parameter this SDK generation exposes: it needs
            # no breakpoint budgeting and cannot be placed wrongly. Fall back to a
            # block-level breakpoint on older SDKs, so that an SDK change downgrades
            # the mechanism rather than silently turning caching off and quadrupling
            # the bill.
            if self._accepts_top_level_cache():
                payload["cache_control"] = self._cache_param()
                self._cache_strategy = f"top_level:{self.cache_ttl}"
            elif self._mark_rolling_breakpoint(payload["messages"]):
                self._cache_strategy = f"rolling_block:{self.cache_ttl}"
            else:
                self._cache_strategy = "requested_but_not_applied"
        else:
            self._cache_strategy = "off"

        return payload

    def _accepts_top_level_cache(self) -> bool:
        declared = declared_parameters(self._client.messages.create)
        # None means the signature could not be read, or takes **kwargs; in either
        # case the top-level parameter is the safer bet.
        return declared is None or "cache_control" in declared

    def _note_dropped(self, name: str) -> None:
        """Record a parameter the SDK or provider refused.

        Caching is tracked separately: if ``cache_control`` is rejected the request
        still succeeds, so without this the run record would keep claiming caching
        was active while quietly billing every token at full rate.
        """
        self._dropped = sorted({*self._dropped, name})
        if name == "cache_control":
            self._cache_strategy = "rejected_by_provider"

    def generate(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSchema],
    ) -> ModelResponse:
        payload = self._payload(system, messages, tools)

        # anthropic 1.8.0 removed temperature and top_p from messages.create, so
        # passing them raises TypeError before any request goes out. Filtering
        # against the installed signature keeps the adapter working across SDK
        # versions instead of pinning one.
        payload, dropped = filter_to_signature(self._client.messages.create, payload)
        for name in dropped:
            self._note_dropped(name)
        self._sampling = describe(payload, self._dropped)

        last_error: Exception | None = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                response = self._client.messages.create(**payload)
            except TypeError as exc:
                # A parameter the signature check could not catch, e.g. one nested
                # inside a TypedDict. Drop it and retry rather than failing the run.
                name = unsupported_param(exc, set(payload))
                if name and name in payload:
                    payload.pop(name)
                    self._note_dropped(name)
                    self._sampling = describe(payload, self._dropped)
                    continue
                return ModelResponse(error=f"{type(exc).__name__}: {exc}")
            except Exception as exc:  # noqa: BLE001 - provider exceptions vary
                name = unsupported_param(exc, set(payload))
                if name and name in payload:
                    payload.pop(name)
                    self._note_dropped(name)
                    self._sampling = describe(payload, self._dropped)
                    continue
                last_error = exc
                if not _is_retryable(exc) or attempt == MAX_ATTEMPTS - 1:
                    return ModelResponse(error=f"{type(exc).__name__}: {exc}")
                time.sleep(BACKOFF_BASE**attempt)
                continue
            return self._parse(response)

        return ModelResponse(error=f"exhausted retries: {last_error}")

    @staticmethod
    def _parse(response: Any) -> ModelResponse:
        texts: list[str] = []
        calls: list[ToolCall] = []
        for block in getattr(response, "content", []) or []:
            kind = getattr(block, "type", None)
            if kind == "text":
                texts.append(getattr(block, "text", ""))
            elif kind == "tool_use":
                calls.append(
                    ToolCall(
                        id=getattr(block, "id", ""),
                        name=getattr(block, "name", ""),
                        arguments=dict(getattr(block, "input", {}) or {}),
                    )
                )

        usage_obj = getattr(response, "usage", None)
        # input_tokens from this API already excludes cache reads and writes, which
        # are reported separately and billed at different rates. Keeping them
        # distinct is what lets pricing.py cost a run correctly rather than
        # charging cached prefix at the full input rate.
        usage = {
            "input_tokens": getattr(usage_obj, "input_tokens", None),
            "output_tokens": getattr(usage_obj, "output_tokens", None),
            "cached_input_tokens": getattr(usage_obj, "cache_read_input_tokens", None),
            "cache_write_tokens": getattr(usage_obj, "cache_creation_input_tokens", None),
        }
        return ModelResponse(
            text="\n".join(t for t in texts if t).strip(),
            tool_calls=calls,
            stop_reason=getattr(response, "stop_reason", None),
            usage={k: v for k, v in usage.items() if v is not None},
        )


def _is_retryable(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    if any(token in name for token in ("ratelimit", "timeout", "connection", "internal", "apistatus")):
        return True
    status = getattr(exc, "status_code", None)
    return status in (408, 409, 429, 500, 502, 503, 504)
