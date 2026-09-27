"""OpenAI Chat Completions adapter.

Spec format: ``openai:<model-id>``, e.g. ``openai:gpt-4.1``.

Requires the ``openai`` package and ``OPENAI_API_KEY``. As with the Anthropic
adapter, the key never reaches the executor environment.
"""

from __future__ import annotations

import json
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
    describe,
    filter_to_signature,
    suggested_fix,
    unsupported_param,
)

MAX_ATTEMPTS = 5
BACKOFF_BASE = 2.0

# Hard ceiling on payload adjustments within a single request, so a provider that
# keeps objecting cannot spin the loop forever.
MAX_REMEDIATION_ROUNDS = 4


class OpenAIModel:
    """Thin adapter over ``client.chat.completions.create``."""

    def __init__(
        self,
        model: str,
        temperature: float = 1.0,
        max_tokens: int = 4096,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.spec = f"openai:{model}"
        self.name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._dropped: list[str] = []
        # Parameters the provider told us to set to a specific value, remembered
        # so they are applied from the first request of every later turn.
        self._forced: dict[str, Any] = {}
        # Remedies already attempted, so the add-rule and the drop-rule cannot
        # undo each other indefinitely.
        self._tried: set[tuple[str, str]] = set()
        self._sampling: dict[str, Any] = {}
        if client is not None:
            self._client = client
        else:
            try:
                import openai
            except ImportError as exc:  # pragma: no cover
                raise ModelError(
                    "the openai package is required for openai:* specs "
                    "(pip install openai)"
                ) from exc
            self._client = openai.OpenAI()

    def describe_sampling(self) -> dict[str, Any]:
        """What the provider actually accepted. Recorded with each run."""
        return dict(self._sampling)

    # -- translation -------------------------------------------------------

    @staticmethod
    def _tools(tools: list[ToolSchema]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]

    @staticmethod
    def _messages(system: str, messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for message in messages:
            if message.role == "assistant":
                entry: dict[str, Any] = {
                    "role": "assistant",
                    "content": message.text or None,
                }
                if message.tool_calls:
                    entry["tool_calls"] = [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments),
                            },
                        }
                        for call in message.tool_calls
                    ]
                out.append(entry)
                continue

            # Tool results are separate messages in this API.
            for result in message.tool_results:
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": result.id,
                        "content": result.content,
                    }
                )
            if message.text:
                out.append({"role": "user", "content": message.text})
        return out

    # -- generation --------------------------------------------------------

    def _payload(
        self, system: str, messages: list[Message], tools: list[ToolSchema]
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            # Newer models reject the older `max_tokens` outright and demand this
            # name instead. Both still exist in the SDK signature, so only the
            # server can tell you which one a given model wants.
            "max_completion_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": self._messages(system, messages),
        }
        if tools:
            payload["tools"] = self._tools(tools)
        return payload

    def generate(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSchema],
    ) -> ModelResponse:
        payload = self._payload(system, messages, tools)
        payload, dropped = filter_to_signature(self._client.chat.completions.create, payload)
        if dropped:
            self._dropped = sorted({*self._dropped, *dropped})
        # Carry forward everything earlier calls established about this model, so a
        # long run does not pay a failed request per turn to relearn it.
        for name in self._dropped:
            payload.pop(name, None)
        payload.update(self._forced)
        self._sampling = describe(payload, self._dropped)

        last_error: Exception | None = None
        attempts = 0
        rounds = 0
        while attempts < MAX_ATTEMPTS:
            attempts += 1
            try:
                response = self._client.chat.completions.create(**payload)
            except Exception as exc:  # noqa: BLE001 - provider exceptions vary
                if self._remediate(exc, payload, rounds):
                    rounds += 1
                    attempts -= 1  # a parameter fix is not a failed attempt
                    continue
                last_error = exc
                if not _is_retryable(exc) or attempts >= MAX_ATTEMPTS:
                    return ModelResponse(error=f"{type(exc).__name__}: {exc}")
                time.sleep(BACKOFF_BASE ** (attempts - 1))
                continue
            return self._parse(response)

        return ModelResponse(error=f"exhausted retries: {last_error}")

    def _remediate(self, exc: Exception, payload: dict[str, Any], rounds: int) -> bool:
        """Adjust the payload in response to a rejection. True if something changed.

        Two kinds of rejection, in priority order:

        1. The error spells out a fix, e.g. gpt-5.6-luna refusing function tools
           unless ``reasoning_effort`` is ``'none'``. That parameter is a
           server-side default, so it has to be *added* — dropping cannot help.
        2. The error names a parameter being sent that the model will not accept,
           e.g. ``max_tokens`` on newer models or ``temperature`` on reasoning
           models. Drop it.

        Each fix is applied once and remembered, so a long run does not spend a
        failed request per turn rediscovering the same constraint. ``_tried`` is
        what stops the two rules fighting: without it, rule 2 can drop what rule 1
        just added, rule 1 re-adds it, and the request never resolves.
        """
        if rounds >= MAX_REMEDIATION_ROUNDS:
            return False

        instruction = suggested_fix(exc)
        if instruction:
            name, value = instruction
            key = (name, repr(value))
            if key not in self._tried:
                payload[name] = value
                self._forced[name] = value
                self._tried.add(key)
                self._sampling = describe(payload, self._dropped)
                return True

        name = unsupported_param(exc, set(payload))
        if (
            name
            and name in payload
            and name not in ("model", "messages", "tools")
            and ("drop", name) not in self._tried
        ):
            payload.pop(name)
            self._dropped = sorted({*self._dropped, name})
            self._forced.pop(name, None)
            self._tried.add(("drop", name))
            self._sampling = describe(payload, self._dropped)
            return True

        return False

    @staticmethod
    def _parse(response: Any) -> ModelResponse:
        choices = getattr(response, "choices", None) or []
        if not choices:
            return ModelResponse(error="provider returned no choices")
        message = choices[0].message

        calls: list[ToolCall] = []
        for call in getattr(message, "tool_calls", None) or []:
            function = getattr(call, "function", None)
            raw_args = getattr(function, "arguments", "") or "{}"
            try:
                arguments = json.loads(raw_args)
            except json.JSONDecodeError:
                arguments = {"__unparsed__": raw_args}
            calls.append(
                ToolCall(
                    id=getattr(call, "id", ""),
                    name=getattr(function, "name", ""),
                    arguments=arguments if isinstance(arguments, dict) else {"value": arguments},
                )
            )

        usage_obj = getattr(response, "usage", None)
        # Unlike Anthropic, prompt_tokens here *includes* cached tokens, so the
        # cached count is subtracted out to leave the portion billed at full rate.
        # Getting this backwards would overstate cost by the full-rate amount on
        # every cache hit.
        cached = getattr(
            getattr(usage_obj, "prompt_tokens_details", None), "cached_tokens", None
        )
        prompt_tokens = getattr(usage_obj, "prompt_tokens", None)
        plain_input = (
            prompt_tokens - cached
            if isinstance(prompt_tokens, int) and isinstance(cached, int)
            else prompt_tokens
        )
        usage = {
            "input_tokens": plain_input,
            "output_tokens": getattr(usage_obj, "completion_tokens", None),
            "cached_input_tokens": cached,
        }
        usage = {k: v for k, v in usage.items() if v is not None}
        return ModelResponse(
            text=(getattr(message, "content", "") or "").strip(),
            tool_calls=calls,
            stop_reason=getattr(choices[0], "finish_reason", None),
            usage=usage,
        )


def _is_retryable(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    if any(token in name for token in ("ratelimit", "timeout", "connection", "internal", "apistatus")):
        return True
    status = getattr(exc, "status_code", None)
    return status in (408, 409, 429, 500, 502, 503, 504)
