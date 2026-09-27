"""Cost and duration estimates for planning a grid.

All per-run figures come from measured runs on this machine, not from guesses:
see MEASURED below. Token figures are summed from trace `usage` records, which
is the only honest way to cost an agent loop — the transcript is re-sent every
turn, so total input tokens are several times the final context length.

Run: python3 tools/estimate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.stats import fisher_exact, required_n  # noqa: E402

# --------------------------------------------------------------------------
# Measured inputs
# --------------------------------------------------------------------------

# Wall-clock seconds per run, max_turns=30. Multiple observations where taken.
MEASURED_SECS = {
    "qwen3.5": {
        "perf_memo": [133.7, 118.0, 163.0],
        "possible_spec": [86.9, 76.8, 149.8],
        "impossible_spec": [617.9, 544.3, 431.0],
    },
    "gemma4": {
        "perf_memo": [85.3],
        "possible_spec": [33.2],
        "impossible_spec": [77.5],
    },
}

# Total tokens per run, summed across turns from trace usage records.
MEASURED_TOKENS = {
    "qwen3.5": {
        "perf_memo": (53_076, 2_174, 7_548),
        "possible_spec": (46_496, 2_387, 6_735),
        "impossible_spec": (199_922, 7_309, 13_015),
    },
    "gemma4": {
        "perf_memo": (17_329, 2_453, 4_160),
        "possible_spec": (10_589, 881, 2_566),
        "impossible_spec": (19_057, 2_222, 4_000),
    },
}

TASKS = ("perf_memo", "possible_spec", "impossible_spec")


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


# --------------------------------------------------------------------------
# Wall-clock for local grids
# --------------------------------------------------------------------------


def local_hours(models: list[str], rungs: int, repeats: int) -> float:
    total = 0.0
    for model in models:
        for task in TASKS:
            total += mean(MEASURED_SECS[model][task]) * rungs * repeats
    return total / 3600


def report_time() -> None:
    print("=" * 74)
    print("LOCAL WALL-CLOCK: 2 models (qwen3.5 + gemma4)")
    print("=" * 74)
    print("mean seconds per run, by model and task:")
    for model in MEASURED_SECS:
        parts = "  ".join(
            f"{t.replace('_spec','').replace('_memo',''):<10}{mean(MEASURED_SECS[model][t]):>6.0f}s"
            for t in TASKS
        )
        per_rung_repeat = sum(mean(MEASURED_SECS[model][t]) for t in TASKS)
        print(f"  {model:<9} {parts}   all three: {per_rung_repeat/60:>5.1f} min")
    print()

    print(f"{'design':<34}{'runs':>7}{'qwen':>9}{'gemma':>9}{'both':>9}")
    print("-" * 74)
    for label, rungs, repeats in (
        ("pilot: P0+P1, n=15", 2, 15),
        ("pilot: P0+P1, n=25", 2, 25),
        ("focused: P0/P1/P3/P6, n=20", 4, 20),
        ("focused: P0/P1/P3/P6, n=35", 4, 35),
        ("full ladder: 7 rungs, n=20", 7, 20),
        ("full ladder: 7 rungs, n=35", 7, 35),
    ):
        runs = 2 * rungs * len(TASKS) * repeats
        q = local_hours(["qwen3.5"], rungs, repeats)
        g = local_hours(["gemma4"], rungs, repeats)
        print(f"{label:<34}{runs:>7}{q:>8.0f}h{g:>8.0f}h{q+g:>8.0f}h")
    print()
    print("Runs per (model, task, rung) cell equals the repeat count; the run total")
    print("is 2 models x rungs x 3 tasks x repeats.")


# --------------------------------------------------------------------------
# Minimum n
# --------------------------------------------------------------------------


def report_power() -> None:
    print()
    print("=" * 74)
    print("MINIMUM RUNS PER CELL")
    print("=" * 74)
    print("Per-arm n for 80% power, alpha=0.05, by baseline rate and absolute change:")
    print(f"{'baseline':<10}" + "".join(f"{f'+/-{d:.2f}':>9}" for d in (0.10, 0.15, 0.20, 0.25, 0.30)))
    print("-" * 74)
    for base in (0.10, 0.20, 0.30, 0.40, 0.50):
        row = f"{base:<10.2f}"
        for delta in (0.10, 0.15, 0.20, 0.25, 0.30):
            other = base + delta
            row += f"{required_n(base, other):>9}" if other < 1.0 else f"{'-':>9}"
        print(row)

    print()
    print("Detecting a rise from a low baseline, which is the P0 -> P1 control:")
    for base, other in ((0.05, 0.25), (0.05, 0.35), (0.10, 0.30), (0.10, 0.40), (0.15, 0.40)):
        print(f"  {base:.2f} -> {other:.2f}   n={required_n(base, other):>4} per arm")

    print()
    print("What a given n can actually show (Fisher, single observed table):")
    for n in (15, 20, 25, 35):
        found = None
        for pct in range(1, 100):
            other = pct / 100
            a, c = round(0.10 * n), round(other * n)
            if other > 0.10 and fisher_exact(a, n - a, c, n - c) < 0.05:
                found = other
                break
        print(f"  n={n:<3} against a 0.10 baseline: needs a rise to ~{found:.2f} to reach p<0.05")


# --------------------------------------------------------------------------
# Cloud cost
# --------------------------------------------------------------------------

# USD per million tokens, standard tier, checked September 2026. Sources are
# listed in the accompanying notes; verify against the provider's own pricing
# page before committing a budget.
CLOUD = {
    "claude-haiku-4.5": (1.0, 5.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-opus-5.5": (4.0, 20.0),
    "gpt-5.6-luna": (0.20, 1.20),
    "gpt-5.6-terra": (2.0, 12.0),
    "gpt-5.6-sol": (5.0, 30.0),
    "gpt-6-astra": (10.0, 50.0),
    # Ollama Cloud, metered. Illustrative small-model rate (gpt-oss:20b class);
    # larger hosted models cost more. Ollama also sells $20/mo and $100/mo tiers
    # that may simply absorb a grid this size.
    "ollama-cloud-20b": (0.07, 0.30),
}

CACHE_READ_FRACTION = 0.10  # cached input bills at ~10% of the input rate
CACHE_WRITE_MULTIPLIER = 1.25  # Anthropic 5-minute cache write premium


def run_cost(
    in_tokens: int,
    out_tokens: int,
    final_ctx: int,
    in_rate: float,
    out_rate: float,
    caching: bool,
) -> float:
    if not caching:
        effective_in = in_tokens
    else:
        # Every token is "new" exactly once, so the unavoidable full-rate portion
        # is about the final context length; the rest is re-sent prefix and bills
        # at the cache-read rate.
        fresh = min(final_ctx, in_tokens)
        reused = in_tokens - fresh
        effective_in = fresh * CACHE_WRITE_MULTIPLIER + reused * CACHE_READ_FRACTION
    return effective_in / 1e6 * in_rate + out_tokens / 1e6 * out_rate


def profile_totals(profile: str) -> tuple[int, int, int]:
    """Total in, out and mean final context for one repeat of all three tasks."""
    data = MEASURED_TOKENS[profile]
    return (
        sum(data[t][0] for t in TASKS),
        sum(data[t][1] for t in TASKS),
        round(mean([data[t][2] for t in TASKS])),
    )


def report_cloud() -> None:
    print()
    print("=" * 74)
    print("CLOUD COST for the same grid shape")
    print("=" * 74)
    rungs, repeats, models = 4, 35, 2
    cells = rungs * len(TASKS) * repeats
    print(f"Shape: focused ladder, {rungs} rungs x 3 tasks x {repeats} repeats")
    print(f"       = {cells} runs per model, {cells*models} runs for two models.")
    print()

    for profile, note in (
        ("gemma4", "efficient: 7-10 turns per run, ~16k input tokens"),
        ("qwen3.5", "verbose: up to 30 turns, ~100k input tokens"),
    ):
        tin, tout, ctx = profile_totals(profile)
        per_run_in, per_run_out = tin / 3, tout / 3
        print(f"-- token profile '{profile}' ({note})")
        print(f"   per run: {per_run_in:,.0f} input, {per_run_out:,.0f} output, "
              f"~{ctx:,} final context")
        print(f"   {'model':<20}{'no caching':>14}{'with caching':>16}{'x2 models':>14}")
        for name, (in_rate, out_rate) in CLOUD.items():
            plain = run_cost(per_run_in, per_run_out, ctx, in_rate, out_rate, False) * cells
            cached = run_cost(per_run_in, per_run_out, ctx, in_rate, out_rate, True) * cells
            print(f"   {name:<20}{'$'+format(plain,',.0f'):>14}"
                  f"{'$'+format(cached,',.0f'):>16}{'$'+format(cached*2,',.0f'):>14}")
        print()

    print("Caching saving on the input side alone:")
    for profile in ("gemma4", "qwen3.5"):
        tin, _tout, ctx = profile_totals(profile)
        per_run_in = tin / 3
        fresh = min(ctx, per_run_in)
        reused = per_run_in - fresh
        effective = fresh * CACHE_WRITE_MULTIPLIER + reused * CACHE_READ_FRACTION
        print(f"  {profile:<9} {reused/per_run_in:>5.0%} of input is re-sent prefix"
              f"  ->  input billed at {effective/per_run_in:>4.0%} of list")


if __name__ == "__main__":
    report_time()
    report_power()
    report_cloud()
