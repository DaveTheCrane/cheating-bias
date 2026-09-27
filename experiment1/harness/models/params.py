"""Defensive request-parameter handling for hosted SDKs.

Two independent things go wrong when calling a provider, and they need different
guards. Both were found by the first paid pilot rather than by reading docs.

1. The installed SDK may not accept a parameter at all. Anthropic 1.8.0 dropped
   ``temperature`` and ``top_p`` from ``messages.create``, so passing them raises
   ``TypeError`` before any request is sent. ``filter_to_signature`` removes
   anything the installed SDK does not declare.

2. The SDK accepts a parameter but the *model* rejects it. OpenAI still declares
   ``max_tokens``, yet newer models return HTTP 400 demanding
   ``max_completion_tokens`` instead, and reasoning models likewise refuse
   ``temperature``. ``unsupported_param`` reads the offending name back out of the
   error so the caller can drop it and retry.

Doing this dynamically rather than hardcoding a model-to-capability table means a
new model release does not silently break a grid, and the adapter reports which
parameters it had to abandon instead of failing.
"""

from __future__ import annotations

import inspect
import json
import re
from typing import Any, Callable

# "Unsupported parameter: 'max_tokens' is not supported with this model."
_QUOTED = re.compile(r"[\"']([A-Za-z_][A-Za-z0-9_.]*)[\"']")

UNSUPPORTED_CODES = (
    "unsupported_parameter",
    "unsupported_value",
    "invalid_parameter",
    "unknown_parameter",
)


def accepts_var_keywords(func: Callable[..., Any]) -> bool:
    """True when the callable takes ``**kwargs`` and so accepts anything."""
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())


def declared_parameters(func: Callable[..., Any]) -> set[str] | None:
    """Names the callable declares, or None when it cannot be determined."""
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return None
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return None
    return {name for name in signature.parameters if name != "self"}


def filter_to_signature(
    func: Callable[..., Any], payload: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """Drop payload keys the installed SDK does not declare.

    Returns:
        ``(accepted, dropped)``. If the signature cannot be inspected, or takes
        ``**kwargs``, the payload passes through untouched.
    """
    declared = declared_parameters(func)
    if declared is None:
        return dict(payload), []

    accepted = {k: v for k, v in payload.items() if k in declared}
    dropped = sorted(set(payload) - set(accepted))
    return accepted, dropped


def _error_payload(exc: Exception) -> dict[str, Any]:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        return body.get("error") if isinstance(body.get("error"), dict) else body
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            data = response.json()
        except Exception:  # noqa: BLE001
            return {}
        if isinstance(data, dict) and isinstance(data.get("error"), dict):
            return data["error"]
    return {}


# "To use function tools, use /v1/responses or set reasoning_effort to 'none'."
_SET_INSTRUCTION = re.compile(
    r"set\s+[\"']?([A-Za-z_][A-Za-z0-9_.]*)[\"']?\s+to\s+[\"']([^\"']+)[\"']",
    re.IGNORECASE,
)

# Values that are really keywords rather than strings.
_LITERALS: dict[str, Any] = {"none": "none", "true": True, "false": False, "null": None}


def suggested_fix(exc: Exception) -> tuple[str, Any] | None:
    """Extract a "set X to 'Y'" remedy that a provider spelled out in its error.

    Some rejections are not about a parameter the caller sent. ``gpt-5.6-luna``
    applies a default ``reasoning_effort`` that is incompatible with function
    tools, and the error names the fix explicitly: set it to ``'none'``. Dropping
    parameters cannot help there, because the offending one was never in the
    payload — it has to be added.

    Returns:
        ``(parameter, value)``, or None when the error suggests nothing actionable.
    """
    error = _error_payload(exc)
    message = str(error.get("message") or exc)

    match = _SET_INSTRUCTION.search(message)
    if not match:
        return None

    name, raw = match.group(1), match.group(2)
    lowered = raw.strip().lower()
    if lowered in _LITERALS and lowered != "none":
        return name, _LITERALS[lowered]
    if raw.strip().isdigit():
        return name, int(raw.strip())
    return name, raw.strip()


def unsupported_param(exc: Exception, candidates: set[str]) -> str | None:
    """Extract the parameter a provider refused, if the error names one.

    Args:
        exc: The provider exception.
        candidates: Parameter names currently being sent. Restricting to these
            avoids acting on an unrelated quoted token in the message.

    Returns:
        The parameter name to drop, or None.
    """
    error = _error_payload(exc)
    code = str(error.get("code") or "")
    named = error.get("param")

    if isinstance(named, str) and named in candidates:
        return named

    message = str(error.get("message") or exc)
    if code and code not in UNSUPPORTED_CODES and "unsupported" not in message.lower():
        return None

    for match in _QUOTED.findall(message):
        if match in candidates:
            return match

    # Fall back to a bare mention, e.g. "temperature is not supported".
    lowered = message.lower()
    for name in sorted(candidates, key=len, reverse=True):
        if f"{name} is not supported" in lowered or f"'{name}'" in lowered:
            return name
    return None


def describe(payload: dict[str, Any], dropped: list[str]) -> dict[str, Any]:
    """Summarise what was actually sent, for the run record.

    Recorded because a config asking for ``temperature: 1.0`` against an SDK that
    no longer accepts it would otherwise leave a result claiming a sampling
    setting that was never applied.
    """
    interesting = (
        "temperature",
        "top_p",
        "max_tokens",
        "max_completion_tokens",
        "output_config",
        "reasoning_effort",
    )
    applied = {k: payload[k] for k in interesting if k in payload}
    return {
        "applied": json.loads(json.dumps(applied, default=str)),
        "dropped": sorted(dropped),
    }
