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

MAX_ATTEMPTS = 5
BACKOFF_BASE = 2.0


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

    def generate(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSchema],
    ) -> ModelResponse:
        last_error: Exception | None = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=self._messages(system, messages),
                    tools=self._tools(tools),
                )
            except Exception as exc:  # noqa: BLE001 - provider exceptions vary
                last_error = exc
                if not _is_retryable(exc) or attempt == MAX_ATTEMPTS - 1:
                    return ModelResponse(error=f"{type(exc).__name__}: {exc}")
                time.sleep(BACKOFF_BASE**attempt)
                continue
            return self._parse(response)

        return ModelResponse(error=f"exhausted retries: {last_error}")

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
