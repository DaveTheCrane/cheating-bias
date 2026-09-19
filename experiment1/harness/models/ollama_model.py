"""Ollama adapter, for locally served models.

Spec format: ``ollama:<model>``, e.g. ``ollama:qwen3.5:latest``. The model id may
itself contain a colon; only the first is treated as the provider separator.

Talks to the native ``/api/chat`` endpoint over the standard library rather than
going through Ollama's OpenAI-compatible shim. Three reasons:

- A local model should not require installing a cloud provider's SDK, nor a
  placeholder API key.
- The native endpoint exposes ``options`` (temperature, seed, num_ctx) and the
  separate ``thinking`` field, which the compatibility layer flattens away.
- Fewer moving parts between the harness and the thing being measured.

Differences from the OpenAI wire format that this adapter absorbs, all confirmed
against Ollama 0.34.0 rather than assumed:

- Tool call ``arguments`` arrive as a decoded object, not a JSON string.
- The system prompt is an ordinary message, not a top-level field.
- Tool results go back as ``{"role": "tool", "tool_name": ..., "content": ...}``;
  the name is recovered by matching the call id.
- Token usage is reported as ``prompt_eval_count`` / ``eval_count``.
- Reasoning models return ``message.thinking`` separately from ``content``.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Callable

from harness.models.base import (
    Message,
    ModelResponse,
    ToolCall,
    ToolSchema,
)

DEFAULT_HOST = "http://127.0.0.1:11434"
MAX_ATTEMPTS = 3
BACKOFF_BASE = 2.0

# Local inference is slow and the first call pays model load time; an observed
# cold start on a 9.7B Q4 model was ~7s, and a 10k-token prompt ~50s. A cloud
# default of 60s would spuriously fail runs here.
DEFAULT_TIMEOUT = 600

# Holds the model resident between runs. Without this, a grid pays the load cost
# on every single run.
DEFAULT_KEEP_ALIVE = "30m"

Transport = Callable[[str, dict[str, Any], int], dict[str, Any]]


def _http_post(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class OllamaModel:
    """Adapter over Ollama's native chat endpoint."""

    def __init__(
        self,
        model: str,
        temperature: float = 1.0,
        max_tokens: int = 4096,
        host: str | None = None,
        think: bool = False,
        num_ctx: int | None = None,
        seed: int | None = None,
        keep_alive: str | None = DEFAULT_KEEP_ALIVE,
        timeout: int = DEFAULT_TIMEOUT,
        transport: Transport | None = None,
    ) -> None:
        """
        Args:
            model: Ollama model tag, e.g. ``qwen3.5:latest``.
            temperature: Sampling temperature.
            max_tokens: Mapped to ``num_predict``.
            host: Base URL. Defaults to ``$OLLAMA_HOST`` then ``127.0.0.1:11434``.
            think: Enable reasoning output on models that support it. The trace
                records it; it is not fed back into the conversation.
            num_ctx: Context window override. Left unset by default, which uses
                the server's own sizing. Set it explicitly if you need runs to be
                comparable across machines, since the server default can differ by
                version and by ``OLLAMA_CONTEXT_LENGTH``.
            seed: Sampling seed, for reproducible single runs.
            keep_alive: How long to hold the model in memory between calls.
            timeout: Per-request timeout in seconds.
            transport: Injection point for tests; defaults to a real HTTP POST.
        """
        self.model = model
        self.spec = f"ollama:{model}"
        self.name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.host = (host or os.environ.get("OLLAMA_HOST") or DEFAULT_HOST).rstrip("/")
        if "://" not in self.host:
            self.host = f"http://{self.host}"
        self.think = think
        self.num_ctx = num_ctx
        self.seed = seed
        self.keep_alive = keep_alive
        self.timeout = timeout
        self._transport: Transport = transport or _http_post
        self._call_counter = 0

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
        out: list[dict[str, Any]] = []
        if system:
            out.append({"role": "system", "content": system})

        # Ollama wants the tool's *name* on each result message. The neutral
        # ToolResult only carries the call id, so build the id -> name map as the
        # assistant turns go by.
        names_by_id: dict[str, str] = {}

        for message in messages:
            if message.role == "assistant":
                entry: dict[str, Any] = {"role": "assistant", "content": message.text or ""}
                if message.tool_calls:
                    entry["tool_calls"] = []
                    for call in message.tool_calls:
                        names_by_id[call.id] = call.name
                        entry["tool_calls"].append(
                            {
                                "id": call.id,
                                "function": {"name": call.name, "arguments": call.arguments},
                            }
                        )
                out.append(entry)
                continue

            for result in message.tool_results:
                block: dict[str, Any] = {"role": "tool", "content": result.content}
                name = names_by_id.get(result.id)
                if name:
                    block["tool_name"] = name
                if result.id:
                    block["tool_call_id"] = result.id
                out.append(block)

            if message.text:
                out.append({"role": "user", "content": message.text})

        return out

    def _options(self) -> dict[str, Any]:
        options: dict[str, Any] = {
            "temperature": self.temperature,
            "num_predict": self.max_tokens,
        }
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
        if self.seed is not None:
            options["seed"] = self.seed
        return options

    def _payload(
        self, system: str, messages: list[Message], tools: list[ToolSchema]
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "think": self.think,
            "messages": self._messages(system, messages),
            "options": self._options(),
        }
        if tools:
            payload["tools"] = self._tools(tools)
        if self.keep_alive:
            payload["keep_alive"] = self.keep_alive
        return payload

    # -- generation --------------------------------------------------------

    def generate(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSchema],
    ) -> ModelResponse:
        payload = self._payload(system, messages, tools)
        url = f"{self.host}/api/chat"
        last_error: str | None = None

        for attempt in range(MAX_ATTEMPTS):
            try:
                raw = self._transport(url, payload, self.timeout)
            except urllib.error.HTTPError as exc:
                detail = _read_error_body(exc)
                # A missing model or a malformed request will not fix itself.
                if exc.code < 500:
                    return ModelResponse(
                        error=f"HTTP {exc.code} from {url}: {detail}"
                        + _hint_for(exc.code, detail, self.model)
                    )
                last_error = f"HTTP {exc.code}: {detail}"
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt == MAX_ATTEMPTS - 1:
                    return ModelResponse(
                        error=(
                            f"could not reach Ollama at {self.host} ({last_error}). "
                            "Is `ollama serve` running?"
                        )
                    )
            except json.JSONDecodeError as exc:
                return ModelResponse(error=f"unparseable response from {url}: {exc}")
            else:
                if isinstance(raw, dict) and raw.get("error"):
                    return ModelResponse(error=f"ollama error: {raw['error']}")
                return self._parse(raw)

            time.sleep(BACKOFF_BASE**attempt)

        return ModelResponse(error=f"exhausted retries against {self.host}: {last_error}")

    def _parse(self, raw: dict[str, Any]) -> ModelResponse:
        message = raw.get("message") or {}

        calls: list[ToolCall] = []
        for entry in message.get("tool_calls") or []:
            function = entry.get("function") or {}
            calls.append(
                ToolCall(
                    id=str(entry.get("id") or self._synthetic_id()),
                    name=str(function.get("name") or ""),
                    arguments=_coerce_arguments(function.get("arguments")),
                )
            )

        return ModelResponse(
            text=(message.get("content") or "").strip(),
            tool_calls=calls,
            thinking=(message.get("thinking") or None),
            stop_reason=raw.get("done_reason"),
            usage={
                "input_tokens": raw.get("prompt_eval_count"),
                "output_tokens": raw.get("eval_count"),
                "total_duration_ms": (
                    round(raw["total_duration"] / 1e6) if raw.get("total_duration") else None
                ),
            },
        )

    def _synthetic_id(self) -> str:
        """Fabricate a call id.

        Ollama 0.34 supplies ids, but older builds did not, and the tool-result
        round trip needs something stable to match on.
        """
        self._call_counter += 1
        return f"ollama-call-{self._call_counter}"


def _coerce_arguments(value: Any) -> dict[str, Any]:
    """Normalise tool arguments to a dict.

    Ollama returns a decoded object, unlike OpenAI's JSON string, but small local
    models sometimes emit a string anyway. Accept both rather than losing the call.
    """
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return {"__unparsed__": value}
        return decoded if isinstance(decoded, dict) else {"value": decoded}
    if value is None:
        return {}
    return {"value": value}


def _read_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return exc.reason or "no detail"
    try:
        return str(json.loads(body).get("error", body))[:400]
    except json.JSONDecodeError:
        return body[:400]


def _hint_for(code: int, detail: str, model: str) -> str:
    if code == 404 or "not found" in detail.lower():
        return f"\nhint: run `ollama pull {model}`, and check `ollama list`."
    if "does not support tools" in detail.lower():
        return (
            "\nhint: this experiment requires a tool-calling model. Check "
            "`ollama show <model>` for a 'tools' capability."
        )
    return ""


def probe(
    model: str,
    host: str | None = None,
    timeout: int = 120,
) -> dict[str, Any]:
    """Check that a local model is reachable and will actually call a tool.

    Returns:
        A dict with ``ok`` plus diagnostic detail. Used by ``harness probe-model``.
    """
    adapter = OllamaModel(model, temperature=0.0, max_tokens=128, host=host, timeout=timeout)
    schema = ToolSchema(
        name="list_dir",
        description="List files in a directory of the working tree.",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Directory path."}},
            "required": [],
        },
    )
    started = time.perf_counter()
    response = adapter.generate(
        "You are working in a checked-out repository. Use the tools available.",
        [Message(role="user", text="List the files in the repository root.")],
        [schema],
    )
    elapsed = time.perf_counter() - started

    return {
        "spec": adapter.spec,
        "host": adapter.host,
        "ok": bool(response.tool_calls) and not response.error,
        "error": response.error,
        "elapsed_secs": round(elapsed, 1),
        "tool_calls": [(call.name, call.arguments) for call in response.tool_calls],
        "text": response.text[:200],
        "usage": response.usage,
    }
