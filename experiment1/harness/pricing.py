"""Token pricing, so every run records what it cost.

Rates are USD per million tokens, standard (non-batch) tier, as researched in
September 2026. They are held here rather than scattered through the code so
there is one place to correct, and every stored run keeps the rate it was priced
at — a later price change therefore cannot silently rewrite the history of what
an old grid cost.

Verify against the provider's own pricing page before trusting a budget. Batch
pricing is deliberately absent: an agent loop is sequential, so each turn depends
on the previous turn's tool results and cannot be submitted as a batch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

RATES_AS_OF = "2026-09"


@dataclass(frozen=True)
class Rate:
    """Per-million-token prices for one model."""

    input_usd: float
    output_usd: float
    cached_input_usd: float | None = None
    cache_write_usd: float | None = None

    def cached_rate(self) -> float:
        """Rate for a cache read, defaulting to the common 10% of input."""
        return self.cached_input_usd if self.cached_input_usd is not None else self.input_usd * 0.10

    def write_rate(self) -> float:
        """Rate for a cache write, defaulting to Anthropic's 1.25x premium."""
        return self.cache_write_usd if self.cache_write_usd is not None else self.input_usd * 1.25


# Matched by longest substring against the model id, so "claude-haiku-4-5-2026..."
# still resolves to the haiku entry.
RATES: dict[str, Rate] = {
    # Anthropic
    "claude-haiku-4.5": Rate(1.0, 5.0, cached_input_usd=0.10),
    "claude-haiku-4-5": Rate(1.0, 5.0, cached_input_usd=0.10),
    "claude-sonnet-5": Rate(2.0, 10.0, cached_input_usd=0.20),
    "claude-opus-5.5": Rate(4.0, 20.0, cached_input_usd=0.20),
    "claude-opus-5": Rate(5.0, 25.0, cached_input_usd=0.50),
    # OpenAI
    "gpt-5.6-luna": Rate(0.20, 1.20, cached_input_usd=0.02),
    "gpt-5.6-terra": Rate(2.0, 12.0, cached_input_usd=0.20),
    "gpt-5.6-sol": Rate(5.0, 30.0, cached_input_usd=0.50),
    "gpt-6-astra": Rate(10.0, 50.0, cached_input_usd=1.0),
    # Local models cost nothing per token; electricity is not billed here.
    "__local__": Rate(0.0, 0.0, cached_input_usd=0.0),
}


def rate_for(spec: str) -> Rate | None:
    """Resolve a rate from a model spec, or None when unpriced.

    Returns None rather than guessing, so an unknown model shows as unpriced in
    the results instead of silently costing zero.
    """
    text = str(spec).lower()
    if text.startswith(("mock:", "ollama:")):
        return RATES["__local__"]

    _provider, _, model_id = text.partition(":")
    best: tuple[int, Rate] | None = None
    for key, rate in RATES.items():
        if key == "__local__":
            continue
        if key in model_id and (best is None or len(key) > best[0]):
            best = (len(key), rate)
    return best[1] if best else None


@dataclass
class Cost:
    """What one run cost, and the tokens behind it."""

    usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    priced: bool = True
    rates_as_of: str = RATES_AS_OF
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "usd": round(self.usd, 6),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "priced": self.priced,
            "rates_as_of": self.rates_as_of,
            **({"detail": self.detail} if self.detail else {}),
        }


def cost_for(spec: str, usage: dict[str, Any]) -> Cost:
    """Price one run from its accumulated usage counters.

    Args:
        spec: The model spec the run used.
        usage: Summed counters. ``input_tokens`` is expected to *exclude* tokens
            billed as cache reads or writes, which providers report separately.

    Returns:
        A Cost. ``priced`` is False when the model has no known rate, in which
        case ``usd`` is zero and should not be treated as free.
    """
    plain_in = int(usage.get("input_tokens") or 0)
    out = int(usage.get("output_tokens") or 0)
    cached_in = int(usage.get("cached_input_tokens") or 0)
    cache_write = int(usage.get("cache_write_tokens") or 0)

    rate = rate_for(spec)
    if rate is None:
        return Cost(
            usd=0.0,
            input_tokens=plain_in,
            output_tokens=out,
            cached_input_tokens=cached_in,
            cache_write_tokens=cache_write,
            priced=False,
        )

    usd = (
        plain_in / 1e6 * rate.input_usd
        + out / 1e6 * rate.output_usd
        + cached_in / 1e6 * rate.cached_rate()
        + cache_write / 1e6 * rate.write_rate()
    )
    return Cost(
        usd=usd,
        input_tokens=plain_in,
        output_tokens=out,
        cached_input_tokens=cached_in,
        cache_write_tokens=cache_write,
        priced=True,
        detail={
            "input_usd_per_mtok": rate.input_usd,
            "output_usd_per_mtok": rate.output_usd,
            "cached_input_usd_per_mtok": rate.cached_rate(),
        },
    )


def project(spec: str, per_run_usage: dict[str, Any], runs: int) -> float:
    """Extrapolate a grid's cost from one run's usage."""
    return cost_for(spec, per_run_usage).usd * runs
