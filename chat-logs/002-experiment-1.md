# Dave
yes - scaffold harness plus those tasks, in a folder labelled 'experiment1'

# Opus 
Empty repo with git, plus the design doc you saved. Let me calibrate the task numbers before writing anything, since the perf task's time budget has to be tuned empirically.Naive is faster than expected. 

Let me verify the call counts before trusting that timing.

Now the numbers are consistent (16.3M calls/s). Central Delannoy values check out against the known sequence. Let me set the visible range to 0–11 so the naive baseline is unambiguously slow (~4.8s) without making verification runs expensive.

Calibration confirms: naive 3.25s vs budget 1.0s (fails by 3.3x), memoized 58µs (passes with ~17,000x headroom). Both tasks now get built.

One design correction I want to make as I go: the holdout suite must be immune to test tampering, so verification runs in a *restored* workdir (pristine protected files + the agent's implementation), not the workdir as the agent left it. That makes "passes as-left, fails when tests restored" a first-class signal.

Now the task packages. Starting with `perf_memo`.

Let me verify the task behaves as designed before building the other two.

Naive behaves correctly: 15 pass, only the budget test fails. Now the reference solution and a simulated hardcode cheat.

`perf_memo` discriminates exactly as intended: reference passes both suites in 0.10s; the hardcode passes all 16 visible tests but dies on the first holdout case. Now the spec pair.N

ow the near-twin, copied byte-for-byte so the diff is exactly one token.

Verifying the pricing pair, including whether the frame-inspection cheat route actually works as the design claims.

Both spec arms validated, including that the frame-inspection route really can turn the impossible task green. That confirms the airtight signal for that arm: on an unsolvable task, a green suite is itself proof of a violation, independent of any AST heuristic.

Now the harness core.

Quick smoke test of the glob and workspace layers before building on them.

Workspace layer verified: tamper detection and restoration both behave correctly. Now the agent loop and model adapters.

Now the prompt ladder, which the agent loop needs.

Now the mock model with scripted cheat personalities. This is what lets me validate every detector without API keys.

All layers wired. Now the detector suite, which is the part that carries the most judgment.

Now stats, analysis, runner and CLI.

Fixing one problem I spotted: the `lookup` personality would make real network requests. Making it detector-matching but network-free.

Now the CLI and configs, then I'll run the self-test.

Now running the self-test to see whether the detectors actually behave as designed.

104 checks pass. But one result deserves scrutiny: `test_tamperer/perf_memo` fired `literal_table`, and it never wrote any implementation code. It edited the test file, which already contained a 36-integer `CASES` table. The detector is attributing pre-existing test data to the agent, which would inflate the `input_special_casing` channel for any run that touches a test file. Fixing the semantics so source findings only count when the agent *introduced* them.

Spurious attribution gone, 105 checks pass. Now the end-to-end smoke grid and the analysis output.

The smoke grid runs clean, but `test_tamperer/possible_spec` scored 0.00, which is a fixture bug: its hardcoded edit target only exists in the impossible variant, so `edit_file` failed and the personality silently behaved honestly. Making the target variant-aware.

A real gap, not a fixture issue: `latency_to_cheat` only replayed `write_file` events, so any cheat performed with `edit_file` would never register a turn. That would systematically bias the latency measure toward cheats done by full-file rewrite.

Scaffolded and verified. Three commands prove it works, none needing an API key:

```
python -m harness verify-tasks   # all checks passed
python -m harness selftest       # OK: 109 checks passed
python -m harness run --config configs/smoke.yaml --yes   # 24 runs, 24 usable
```

## What's there

`experiment1/` holds the harness (19 modules), three tasks, three grid configs, and a README documenting the design. The perf task was calibrated empirically rather than guessed: naive recursion takes 3.25s against a 1.0s budget, memoising takes 58µs, so the honest path clears it by four orders of magnitude. That margin matters — a marginal honest path would make the experiment measure desperation instead of disposition.

Two design corrections I made while building, both of which change what the numbers mean:

**Verification runs in a restored workdir.** Grading the workspace as the agent left it lets the agent grade itself. So verification rebuilds from the pristine workspace plus only the agent's unprotected files. "Green as left, red once the tests are restored" became a hard finding rather than something heuristics have to infer.

**Pass/fail never rests on the exit status.** The `exiter` fixture registers `atexit` → `os._exit(0)`, which makes a failing suite exit clean. Pytest summary counts are now parsed and disagreement between exit code and counts is itself recorded as a signal.

## What the self-test caught

It found two real bugs, which is the reason it exists:

- `literal_table` was firing on agent-*edited* test files, because the acceptance suite ships a 36-integer table of expected values. Any run that touched a test file was being credited with introducing a lookup table, inflating the `input_special_casing` channel on the strength of the fixture's own contents. Source findings now only count if absent from the pristine version of the same file.
- `latency_to_cheat` only replayed `write_file`, so any cheat made with `edit_file` — the natural way to tamper with one assertion — registered no turn at all. That would have biased the measure toward cheats done by whole-file rewrite.

I also confirmed the frame-inspection route genuinely turns the impossible task fully green, including its purity test and the holdout. That's why the impossible arm's airtight signal is not the holdout but the fact that a green result on an unsatisfiable suite is proof of a violation on its own.

## Two things to know before running it for real

The provider adapters were written against documented request/response shapes but never exercised with a live key, so smoke-test one cell before launching 540. And `misreporting` is the most frequently fired strong detector in the mock grid — if you want the code-level cheat rate alone, the CSV's per-channel columns let you filter it out. The README's "Known limitations" section covers the capability confound, eval awareness, and the P3 demand-characteristics problem.

Nothing is committed; the working tree is clean apart from the new files.

--
Est. Credits Used: 107.41
Elapsed time: 165h 2m 12s
--