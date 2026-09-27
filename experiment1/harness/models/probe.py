"""One-shot reachability and tool-calling check for any model spec.

Worth running before every grid, and especially before the first paid one. Three
failures this catches for the price of a few hundred tokens:

- A wrong model id. The probe lists what the account can actually see instead of
  leaving a 404 to interpret.
- A model that will not emit tool calls. Those runs look like total task failure,
  which reads as a capability result rather than a plumbing problem.
- An adapter bug. The hosted adapters were written against documented
  request/response shapes; this is the cheapest way to exercise one for real.
"""

from __future__ import annotations

import time
from typing import Any

from harness.models import build_model, parse_spec
from harness.models.base import Message, ToolSchema
from harness.pricing import cost_for

PROBE_TOOL = ToolSchema(
    name="list_dir",
    description="List files in a directory of the working tree.",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Directory path."}},
        "required": [],
    },
)

PROBE_SYSTEM = "You are working in a checked-out repository. Use the tools available."
PROBE_USER = "List the files in the repository root."


def probe(spec: str, provider_options: dict[str, dict] | None = None) -> dict[str, Any]:
    """Issue a single tool-calling request and report what came back."""
    provider, model_id = parse_spec(spec)

    try:
        model = build_model(
            spec, temperature=0.0, max_tokens=256, provider_options=provider_options
        )
    except Exception as exc:  # noqa: BLE001 - surface construction failures plainly
        return {
            "spec": spec,
            "provider": provider,
            "ok": False,
            "error": f"could not construct the adapter: {type(exc).__name__}: {exc}",
            "available_models": [],
        }

    started = time.perf_counter()
    response = model.generate(PROBE_SYSTEM, [Message(role="user", text=PROBE_USER)], [PROBE_TOOL])
    elapsed = time.perf_counter() - started

    cost = cost_for(spec, response.usage or {})
    report: dict[str, Any] = {
        "spec": spec,
        "provider": provider,
        "model_id": model_id,
        "ok": bool(response.tool_calls) and not response.error,
        "error": response.error,
        "elapsed_secs": round(elapsed, 1),
        "tool_calls": [(c.name, c.arguments) for c in response.tool_calls],
        "text": (response.text or "")[:200],
        "thinking": bool(response.thinking),
        "usage": response.usage or {},
        "cost_usd": cost.usd,
        "priced": cost.priced,
        "available_models": [],
    }

    if report["error"]:
        report["available_models"] = list_models(provider)

    return report


def list_models(provider: str, limit: int = 40) -> list[str]:
    """Best-effort list of model ids the credentials can see.

    Only called after a failure, to turn a bad model id into an obvious one.
    """
    try:
        if provider == "anthropic":
            import anthropic

            return [m.id for m in anthropic.Anthropic().models.list(limit=limit).data]
        if provider == "openai":
            import openai

            return sorted(m.id for m in openai.OpenAI().models.list().data)[:limit]
    except Exception:  # noqa: BLE001 - diagnostics must not raise
        return []
    return []
