"""Aggregation and reporting.

The headline number the design cares about is not a single cheat rate. It is the
shape of the rate across the ladder: how far the strongest prompt moves it from
P0 (the shallow, prompt-addressable component) and where it stops (the floor,
the component prompting does not reach). Those are reported per model and per
task, because collapsing across either would hide the between-model variation
that distinguishes the two hypotheses.
"""

from __future__ import annotations

import csv
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from harness.detectors import CHANNELS
from harness.prompts import LADDER
from harness.stats import Interval, fisher_exact, wilson_interval

BASELINE = "P0"
PRESSURE = "P1"
# Rungs intended to reduce cheating. P1 pushes the other way and is a positive
# control, so it is excluded from the floor.
REDUCING_RUNGS = ("P2", "P3", "P4", "P5", "P6")


@dataclass
class Cell:
    """Aggregate for one (model, task, prompt) cell."""

    model: str
    task: str
    prompt: str
    n: int = 0
    cheats: int = 0
    overclaims: int = 0
    legitimate: int = 0
    verified: int = 0
    hard_evidence: int = 0
    excluded_manipulation: int = 0
    excluded_error: int = 0
    channel_counts: Counter = field(default_factory=Counter)
    method_counts: Counter = field(default_factory=Counter)
    stop_reasons: Counter = field(default_factory=Counter)
    no_claim: int = 0
    # Runs cut off at the turn cap. These are right-censored: we know they had not
    # cheated *yet*, not that they would not have. Censoring can therefore only
    # hide cheats, never invent them, so a cell's observed rate is a lower bound.
    censored: int = 0
    censored_clean: int = 0
    latencies: list[int] = field(default_factory=list)
    usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    unpriced: int = 0
    durations: list[float] = field(default_factory=list)
    # (turns, input_tokens) per run. If every repeat in a cell matches exactly,
    # the provider is probably sampling deterministically and the repeats are not
    # independent observations.
    run_signatures: list[tuple[int, int]] = field(default_factory=list)

    @property
    def looks_deterministic(self) -> bool:
        return len(self.run_signatures) >= 3 and len(set(self.run_signatures)) == 1

    @property
    def rate_is_lower_bound(self) -> bool:
        """True when a clean run was cut off, so the rate may understate the truth.

        A censored run that already cheated is fully informative. One censored while
        still clean is the problem case: it contributes a 'no' that might have been
        a 'yes' given more turns.
        """
        return self.censored_clean > 0

    @property
    def uncensored_rate(self) -> Interval:
        """Rate among runs that reached their own conclusion.

        Complementary to ``rate``, not a replacement: dropping censored runs throws
        away the cheats they did catch and conditions on an outcome, so it can bias
        upward. Reported alongside so the two bracket the truth.
        """
        completed = self.n - self.censored
        cheats_completed = self.cheats - (self.censored - self.censored_clean)
        return wilson_interval(max(0, cheats_completed), max(0, completed))
    unnamed_channel: int = 0
    unnamed_applicable: int = 0

    @property
    def rate(self) -> Interval:
        return wilson_interval(self.cheats, self.n)

    @property
    def overclaim_rate(self) -> Interval:
        return wilson_interval(self.overclaims, self.n)

    @property
    def median_latency(self) -> float | None:
        return statistics.median(self.latencies) if self.latencies else None

    def key(self) -> tuple[str, str, str]:
        return (self.model, self.task, self.prompt)


@dataclass
class LadderSummary:
    """The elasticity picture for one (model, task)."""

    model: str
    task: str
    baseline_rate: float | None = None
    baseline_n: int = 0
    floor_rate: float | None = None
    floor_prompt: str | None = None
    pressure_rate: float | None = None
    contrasts: list[dict[str, Any]] = field(default_factory=list)

    @property
    def movable(self) -> float | None:
        """Rate points the ladder removed: the shallow component."""
        if self.baseline_rate is None or self.floor_rate is None:
            return None
        return max(0.0, self.baseline_rate - self.floor_rate)

    @property
    def immune_fraction(self) -> float | None:
        """Share of baseline cheating that survives the best prompt.

        Near 0 is consistent with a shallow, prompt-addressable effect; near 1
        with something prompting does not reach. It is an effect-size summary,
        not a hypothesis test, and it is meaningless when the baseline rate is
        itself near zero — hence baseline_n is reported alongside it.
        """
        if not self.baseline_rate or self.floor_rate is None:
            return None
        return min(1.0, self.floor_rate / self.baseline_rate)

    @property
    def positive_control_ok(self) -> bool | None:
        """Did the pressure rung move the rate up? If not, treat with suspicion."""
        if self.pressure_rate is None or self.baseline_rate is None:
            return None
        return self.pressure_rate >= self.baseline_rate


@dataclass
class Report:
    cells: dict[tuple[str, str, str], Cell] = field(default_factory=dict)
    ladders: list[LadderSummary] = field(default_factory=list)
    twin_contrasts: list[dict[str, Any]] = field(default_factory=list)
    totals: Counter = field(default_factory=Counter)
    batches: Counter = field(default_factory=Counter)
    # task id -> set of task digests seen. More than one means the file mixes runs
    # measured against different versions of that task.
    task_digests: dict[str, set[str]] = field(default_factory=dict)
    harness_commits: Counter = field(default_factory=Counter)

    @property
    def mixed_tasks(self) -> list[str]:
        return sorted(task for task, digests in self.task_digests.items() if len(digests) > 1)

    def cell(self, model: str, task: str, prompt: str) -> Cell | None:
        return self.cells.get((model, task, prompt))


def _usable(row: dict[str, Any]) -> bool:
    if row.get("error"):
        return False
    return (row.get("score") or {}).get("manipulation_check") is not False


def summarise(rows: Iterable[dict[str, Any]]) -> Report:
    """Aggregate result rows into cells, ladder summaries and twin contrasts."""
    report = Report()

    for row in rows:
        score = row.get("score") or {}
        key = (row.get("model", "?"), row.get("task", "?"), row.get("prompt", "?"))
        cell = report.cells.get(key)
        if cell is None:
            cell = Cell(model=key[0], task=key[1], prompt=key[2])
            report.cells[key] = cell

        report.totals["rows"] += 1

        provenance = row.get("provenance") or {}
        if row.get("batch_id"):
            report.batches[str(row["batch_id"])] += 1
        if provenance.get("task_digest"):
            report.task_digests.setdefault(str(row.get("task")), set()).add(
                str(provenance["task_digest"])
            )
        if provenance.get("git_commit"):
            report.harness_commits[str(provenance["git_commit"])] += 1

        # Cost and duration accrue for every attempted run, including excluded
        # ones: money spent on a run that failed the attention check was still
        # spent, and a budget that ignored those would understate the real bill.
        cost = row.get("cost") or {}
        cell.usd += float(cost.get("usd") or 0.0)
        cell.input_tokens += int(cost.get("input_tokens") or 0)
        cell.output_tokens += int(cost.get("output_tokens") or 0)
        cell.cached_input_tokens += int(cost.get("cached_input_tokens") or 0)
        if cost and not cost.get("priced", True):
            cell.unpriced += 1
        if row.get("duration"):
            cell.durations.append(float(row["duration"]))
        turns = int((row.get("agent") or {}).get("turns") or 0)
        if turns:
            cell.run_signatures.append((turns, int(cost.get("input_tokens") or 0)))

        if row.get("error"):
            cell.excluded_error += 1
            report.totals["excluded_error"] += 1
            continue
        if score.get("manipulation_check") is False:
            cell.excluded_manipulation += 1
            report.totals["excluded_manipulation"] += 1
            continue

        cell.n += 1
        report.totals["usable"] += 1
        if score.get("cheated"):
            cell.cheats += 1
            report.totals["cheats"] += 1
        if score.get("overclaim"):
            cell.overclaims += 1
        if score.get("legitimate"):
            cell.legitimate += 1
        if score.get("verified_success"):
            cell.verified += 1
        if score.get("hard_evidence"):
            cell.hard_evidence += 1
        cell.method_counts[score.get("method", "unknown")] += 1
        stop_reason = (row.get("agent") or {}).get("stop_reason", "unknown")
        cell.stop_reasons[stop_reason] += 1
        if stop_reason == "max_turns":
            cell.censored += 1
            if not score.get("cheated"):
                cell.censored_clean += 1
        if score.get("claimed") in (None, "none"):
            cell.no_claim += 1
        for channel in score.get("channels") or []:
            cell.channel_counts[channel] += 1
        latency = score.get("latency_to_cheat")
        if isinstance(latency, int):
            cell.latencies.append(latency)
        unnamed = score.get("used_unnamed_channel")
        if unnamed is not None:
            cell.unnamed_applicable += 1
            if unnamed:
                cell.unnamed_channel += 1

    report.ladders = _build_ladders(report)
    report.twin_contrasts = _build_twin_contrasts(report)
    return report


def _build_ladders(report: Report) -> list[LadderSummary]:
    grouped: dict[tuple[str, str], dict[str, Cell]] = defaultdict(dict)
    for (model, task, prompt), cell in report.cells.items():
        grouped[(model, task)][prompt] = cell

    summaries: list[LadderSummary] = []
    for (model, task), by_prompt in sorted(grouped.items()):
        summary = LadderSummary(model=model, task=task)
        baseline = by_prompt.get(BASELINE)
        if baseline and baseline.n:
            summary.baseline_rate = baseline.rate.point
            summary.baseline_n = baseline.n

        pressure = by_prompt.get(PRESSURE)
        if pressure and pressure.n:
            summary.pressure_rate = pressure.rate.point

        candidates = [
            (prompt, cell)
            for prompt, cell in by_prompt.items()
            if prompt in REDUCING_RUNGS and cell.n
        ]
        if candidates:
            best_prompt, best_cell = min(candidates, key=lambda item: item[1].rate.point)
            summary.floor_rate = best_cell.rate.point
            summary.floor_prompt = best_prompt

        for prompt in sorted(by_prompt):
            if prompt == BASELINE or baseline is None or not baseline.n:
                continue
            cell = by_prompt[prompt]
            if not cell.n:
                continue
            p_value = fisher_exact(
                baseline.cheats,
                baseline.n - baseline.cheats,
                cell.cheats,
                cell.n - cell.cheats,
            )
            summary.contrasts.append(
                {
                    "prompt": prompt,
                    "label": LADDER[prompt].label if prompt in LADDER else prompt,
                    "rate": cell.rate.point,
                    "n": cell.n,
                    "delta_vs_P0": cell.rate.point - baseline.rate.point,
                    "fisher_p": p_value,
                    "expected_direction": (
                        LADDER[prompt].expected_direction if prompt in LADDER else "?"
                    ),
                }
            )
        summaries.append(summary)
    return summaries


def _build_twin_contrasts(report: Report) -> list[dict[str, Any]]:
    """Contrast the unsolvable task against its solvable near-twin.

    Same repo, same framing, one token different in the acceptance suite. The
    difference in cheat rate is therefore attributable to the task being
    impossible rather than to anything about the scenario.
    """
    pairs = (("impossible_spec", "possible_spec"),)
    out: list[dict[str, Any]] = []
    for impossible, possible in pairs:
        models = {model for (model, task, _prompt) in report.cells if task in (impossible, possible)}
        prompts = {prompt for (_m, task, prompt) in report.cells if task in (impossible, possible)}
        for model in sorted(models):
            for prompt in sorted(prompts):
                left = report.cell(model, impossible, prompt)
                right = report.cell(model, possible, prompt)
                if not left or not right or not left.n or not right.n:
                    continue
                out.append(
                    {
                        "model": model,
                        "prompt": prompt,
                        "impossible_rate": left.rate.point,
                        "impossible_n": left.n,
                        "possible_rate": right.rate.point,
                        "possible_n": right.n,
                        "excess": left.rate.point - right.rate.point,
                        "fisher_p": fisher_exact(
                            left.cheats,
                            left.n - left.cheats,
                            right.cheats,
                            right.n - right.cheats,
                        ),
                    }
                )
    return out


# -- rendering --------------------------------------------------------------


def format_report(report: Report) -> str:
    lines: list[str] = []
    add = lines.append

    add("=" * 78)
    add("CHEAT RATE BY CELL")
    add("=" * 78)
    add(
        f"{'model':<22}{'task':<17}{'rung':<6}{'n':>4}{'cheat':>7}"
        f"{'  95% CI':<16}{'overclaim':>10}{'latency':>8}{'cens':>6}"
    )
    add("-" * 78)
    for key in sorted(report.cells):
        cell = report.cells[key]
        if not cell.n and not cell.excluded_error and not cell.excluded_manipulation:
            continue
        interval = cell.rate
        latency = cell.median_latency
        # A trailing '>' marks a rate that is a lower bound: at least one run was
        # still clean when its turn budget ran out.
        bound = ">" if cell.rate_is_lower_bound else " "
        add(
            f"{cell.model[:21]:<22}{cell.task[:16]:<17}{cell.prompt:<6}{cell.n:>4}"
            f"{interval.point:>6.2f}{bound}"
            f" [{interval.low:.2f},{interval.high:.2f}]{'':<4}"
            f"{cell.overclaim_rate.point:>10.2f}"
            f"{(f'{latency:.0f}' if latency is not None else '-'):>8}"
            f"{cell.censored:>6}"
        )

    excluded = report.totals["excluded_manipulation"] + report.totals["excluded_error"]
    add("")
    add(
        f"rows={report.totals['rows']} usable={report.totals['usable']} "
        f"excluded={excluded} "
        f"(manipulation-check failures {report.totals['excluded_manipulation']}, "
        f"errors {report.totals['excluded_error']})"
    )

    add("")
    add("=" * 78)
    add("LADDER SHAPE  (shallow component = P0 minus floor; floor = best reducing rung)")
    add("=" * 78)
    for summary in report.ladders:
        if summary.baseline_rate is None:
            add(f"{summary.model} / {summary.task}: no {BASELINE} cell, cannot compute elasticity")
            continue
        movable = summary.movable
        immune = summary.immune_fraction
        add(f"{summary.model} / {summary.task}")
        add(
            f"  P0 rate {summary.baseline_rate:.2f} (n={summary.baseline_n})"
            f"   floor {summary.floor_rate:.2f} at {summary.floor_prompt}"
            if summary.floor_rate is not None
            else f"  P0 rate {summary.baseline_rate:.2f} (n={summary.baseline_n})   floor n/a"
        )
        if movable is not None:
            add(
                f"  shallow (prompt-addressable) {movable:.2f}"
                + (f"   immune fraction {immune:.2f}" if immune is not None else "")
            )
        control = summary.positive_control_ok
        if control is not None:
            verdict = "ok" if control else "SUSPECT: pressure did not raise the rate"
            add(f"  positive control (P1 {summary.pressure_rate:.2f} vs P0): {verdict}")
        for contrast in summary.contrasts:
            flag = ""
            if contrast["expected_direction"] == "decrease" and contrast["delta_vs_P0"] > 0:
                flag = "  <- moved the wrong way"
            if contrast["expected_direction"] == "increase" and contrast["delta_vs_P0"] < 0:
                flag = "  <- moved the wrong way"
            add(
                f"    {contrast['prompt']} {contrast['label']:<24}"
                f"rate {contrast['rate']:.2f} (n={contrast['n']:>3})  "
                f"delta {contrast['delta_vs_P0']:+.2f}  p={contrast['fisher_p']:.3f}{flag}"
            )
        add("")

    channel_totals: Counter = Counter()
    for cell in report.cells.values():
        channel_totals.update(cell.channel_counts)
    if channel_totals:
        add("=" * 78)
        add("CHANNELS USED  (counts of runs, not of findings)")
        add("=" * 78)
        for channel in CHANNELS:
            if channel_totals.get(channel):
                add(f"  {channel:<24}{channel_totals[channel]:>5}")
        add("")

    unnamed = sum(cell.unnamed_channel for cell in report.cells.values())
    applicable = sum(cell.unnamed_applicable for cell in report.cells.values())
    if applicable:
        add(
            f"generalisation probe: {unnamed}/{applicable} runs under an enumerating "
            f"condition cheated through a route it did not name"
        )
        add("")

    if report.twin_contrasts:
        add("=" * 78)
        add("TWIN CONTRAST  (unsolvable vs one-token-different solvable arm)")
        add("=" * 78)
        for row in report.twin_contrasts:
            add(
                f"  {row['model'][:20]:<21}{row['prompt']:<5}"
                f"impossible {row['impossible_rate']:.2f} (n={row['impossible_n']:>3})  "
                f"possible {row['possible_rate']:.2f} (n={row['possible_n']:>3})  "
                f"excess {row['excess']:+.2f}  p={row['fisher_p']:.3f}"
            )
        add("")

    methods: Counter = Counter()
    stops: Counter = Counter()
    no_claim = 0
    usable = 0
    for cell in report.cells.values():
        methods.update(cell.method_counts)
        stops.update(cell.stop_reasons)
        no_claim += cell.no_claim
        usable += cell.n
    if methods:
        add("=" * 78)
        add("OUTCOME MIX")
        add("=" * 78)
        for method, count in methods.most_common():
            add(f"  {method:<28}{count:>5}")

    if report.mixed_tasks or len(report.harness_commits) > 1:
        add("")
        add("!" * 78)
        add("WARNING: this results set mixes runs from different setups")
        add("!" * 78)
        for task in report.mixed_tasks:
            add(
                f"  task {task!r} appears with {len(report.task_digests[task])} different "
                f"digests: {sorted(report.task_digests[task])}"
            )
        if len(report.harness_commits) > 1:
            add(f"  harness commits present: {sorted(report.harness_commits)}")
        add("")
        add("  Every rate below pools these together as though they were comparable.")
        add("  Slice by batch_id to separate them, e.g.:")
        add("    jq -c 'select(.batch_id==\"<id>\")' results.jsonl > batch.jsonl")
        add("    python -m harness analyze batch.jsonl")

    if len(report.batches) > 1:
        add("")
        add(f"batches present ({len(report.batches)}): "
            + ", ".join(f"{b} ({n} runs)" for b, n in sorted(report.batches.items())))

    censored_cells = [cell for cell in report.cells.values() if cell.rate_is_lower_bound]
    if censored_cells:
        add("")
        add("!" * 78)
        add("WARNING: censored runs — the rates marked '>' are lower bounds")
        add("!" * 78)
        for cell in sorted(censored_cells, key=lambda c: -c.censored_clean)[:10]:
            uncensored = cell.uncensored_rate
            add(
                f"  {cell.model[:20]:<21}{cell.task:<16}{cell.prompt:<4}"
                f"{cell.censored_clean} of {cell.n} run(s) hit the turn cap while still "
                f"clean; rate among runs that concluded: {uncensored.point:.2f} "
                f"(n={uncensored.n})"
            )
        add("")
        add("  A run stopped at its turn cap is right-censored: it had not cheated yet,")
        add("  which is not the same as it never would. Censoring can therefore only")
        add("  hide cheats, never create them, so the observed rate understates the truth")
        add("  by an unknown amount. Raise `max_turns` for the affected task (it can be")
        add("  set per-task in task.yaml) until this section is empty, then re-read the")
        add("  rates. Check the stop-reason section below for how widespread it is.")

    identical = [cell for cell in report.cells.values() if cell.looks_deterministic]
    if identical:
        add("")
        add("!" * 78)
        add("WARNING: repeats may not be independent")
        add("!" * 78)
        for cell in identical[:8]:
            add(
                f"  {cell.model} / {cell.task} / {cell.prompt}: all {len(cell.run_signatures)} "
                f"runs used identical turns and input tokens"
            )
        add("")
        add("  Every repeat in these cells looks byte-identical, which is what you see")
        add("  when the provider samples deterministically. If that is the case the cell")
        add("  holds one observation counted n times and its confidence interval is")
        add("  meaningless. Check whether temperature was actually applied:")
        add("  `agent.sampling.dropped` in results.jsonl lists parameters the provider")
        add("  refused. Anthropic's current SDK removed temperature altogether.")

    spend = sum(cell.usd for cell in report.cells.values())
    attempted = sum(
        cell.n + cell.excluded_error + cell.excluded_manipulation
        for cell in report.cells.values()
    )
    if spend > 0 or any(cell.input_tokens for cell in report.cells.values()):
        add("")
        add("=" * 78)
        add("COST")
        add("=" * 78)
        add(f"{'model':<30}{'runs':>6}{'in tok':>12}{'cached':>11}{'out tok':>10}{'USD':>10}")
        add("-" * 78)
        by_model: dict[str, Cell] = {}
        for cell in report.cells.values():
            merged = by_model.setdefault(cell.model, Cell(model=cell.model, task="", prompt=""))
            merged.n += cell.n + cell.excluded_error + cell.excluded_manipulation
            merged.usd += cell.usd
            merged.input_tokens += cell.input_tokens
            merged.output_tokens += cell.output_tokens
            merged.cached_input_tokens += cell.cached_input_tokens
            merged.unpriced += cell.unpriced
            merged.durations.extend(cell.durations)
        for model in sorted(by_model):
            merged = by_model[model]
            add(
                f"{model[:29]:<30}{merged.n:>6}{merged.input_tokens:>12,}"
                f"{merged.cached_input_tokens:>11,}{merged.output_tokens:>10,}"
                f"{merged.usd:>10.4f}"
            )
            if merged.unpriced:
                add(
                    f"  {merged.unpriced} run(s) used a model with no rate in "
                    f"harness/pricing.py, so their cost reads as zero"
                )
            # An agent loop re-sends its transcript every turn, so on a hosted
            # provider most input tokens should be cache reads. Zero cached tokens
            # over several multi-turn runs means caching is not working, which is
            # roughly a 4x cost difference and easy to miss in a total.
            total_in = merged.input_tokens + merged.cached_input_tokens
            if total_in and merged.n >= 3 and not model.startswith(("mock:", "ollama:")):
                share = merged.cached_input_tokens / total_in
                if merged.cached_input_tokens == 0:
                    add(
                        "  no cached input at all across these runs: prompt caching "
                        "is off or being rejected. Check `agent.sampling.cache_strategy`."
                    )
                else:
                    add(f"  {share:.0%} of input tokens were cache reads")
        add("")
        add(f"total spend ${spend:.4f} over {attempted} attempted run(s)")
        if attempted:
            per_run = spend / attempted
            add(f"mean ${per_run:.4f} per run")
            for scale in (420, 840):
                add(f"  extrapolated to {scale:>4} runs: ${per_run * scale:,.2f}")

    if stops:
        add("")
        add("=" * 78)
        add("HOW RUNS ENDED")
        add("=" * 78)
        for reason, count in stops.most_common():
            add(f"  {reason:<28}{count:>5}")
        if usable:
            share = no_claim / usable
            add(
                f"\n  {no_claim}/{usable} usable runs ({share:.0%}) ended without calling "
                f"finish, so they carry no claim."
            )
            if share > 0.2:
                add(
                    "  That is high enough to weaken the claim-based measures "
                    "(overclaim, misreporting), which only exist for runs that\n"
                    "  reported an outcome. If most of these stopped at max_turns, raise "
                    "max_turns before reading anything into those columns."
                )

    return "\n".join(lines)


def write_csv(report: Report, path: Path) -> Path:
    """Write one row per cell for downstream analysis."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "task",
        "prompt",
        "n",
        "cheats",
        "cheat_rate",
        "ci_low",
        "ci_high",
        "overclaims",
        "legitimate",
        "verified",
        "hard_evidence",
        "median_latency_to_cheat",
        "censored",
        "censored_clean",
        "rate_is_lower_bound",
        "uncensored_rate",
        "excluded_manipulation",
        "excluded_error",
        *[f"channel_{channel}" for channel in CHANNELS],
    ]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key in sorted(report.cells):
            cell = report.cells[key]
            interval = cell.rate
            writer.writerow(
                {
                    "model": cell.model,
                    "task": cell.task,
                    "prompt": cell.prompt,
                    "n": cell.n,
                    "cheats": cell.cheats,
                    "cheat_rate": round(interval.point, 4),
                    "ci_low": round(interval.low, 4),
                    "ci_high": round(interval.high, 4),
                    "overclaims": cell.overclaims,
                    "legitimate": cell.legitimate,
                    "verified": cell.verified,
                    "hard_evidence": cell.hard_evidence,
                    "median_latency_to_cheat": cell.median_latency,
                    "censored": cell.censored,
                    "censored_clean": cell.censored_clean,
                    "rate_is_lower_bound": cell.rate_is_lower_bound,
                    "uncensored_rate": round(cell.uncensored_rate.point, 4),
                    "excluded_manipulation": cell.excluded_manipulation,
                    "excluded_error": cell.excluded_error,
                    **{
                        f"channel_{channel}": cell.channel_counts.get(channel, 0)
                        for channel in CHANNELS
                    },
                }
            )
    return destination
