# Pillar A wrap-up handoff

Date: November 9, 2026. Baseline: the submitted October 19 Progress Report 3.
This is implementation/evaluation documentation, not an academic report.

The revised scope is a bounded CPU shared-prefix KV research prototype. Pillars B
(tool scheduling) and C (grammar masking) remain deferred. The three-week-sized
work block is organized as correctness/lifecycle closure, repeated measurement,
and reproducible evaluation; it does not claim three weeks of elapsed labor.
See [acceptance scope](../../docs/PILLAR_A_ACCEPTANCE.md) and
[demo commands](../../docs/PILLAR_A_DEMO.md).

## Implementation completed

1. Internal-prefix forks now share positions [0, prefix depth), rather than a
   descendant's entire physical sequence. A fork can therefore use a longer
   descendant as its source without inheriting the descendant's suffix.
2. If a backend share throws, its reserved sequence slot is returned before the
   exception propagates; reference counts are not retained on that failure.
3. An explicitly requested occupied slot now fails instead of silently selecting
   another slot whose identity the caller was never told.
4. Capacity discovery advances by the pinned backend's effective allocation
   quantum: 256 for unified storage, b*256 for independent streams. This avoids
   repeatedly executing workloads at requests that round to the same capacity.
5. `--teacher-forced` diagnostic replay keeps branch-specific input histories
   identical in all compared executions, so numerical comparisons cover every
   step even after predictions differ. It requires a verification flag and is
   excluded from the performance dataset. The 100% tokens / 1e-3 max-logit
   requirement was not relaxed.
6. The attention-ablation runner now requires batched parity before recording
   performance. Failed parity stops that ablation and retains its logs.
7. The CPU sampler initializes without racing the child's loader, refreshes its
   modules after a phase marker, resolves addresses live and labels nearest DLL
   exports separately from PDB symbols. Flat-profile charts retain full names in
   CSV but shorten display labels. Performance error bars now show actual Q1-Q3,
   with analysis script/input hashes recorded separately.

No vendored backend source or submodule pin was changed. All real-model runs use
pinned llama.cpp `434ddbbc0e30522e897670681e503b797c12b7c1`.

## Tests and numerical investigation

- **21 internal tree tests pass**, including three new regressions that failed
  before the fixes. Their failing output is preserved in `validation/`.
- **5/5 real-enabled CTest targets pass**; a fresh mock-only build passes **2/2**.
- The actual-model lifecycle test completes **12 eviction/reuse cycles**. It
  checks prefix-copy position bounds, different suffix inputs, committed-path
  reachability, release/eviction, slot reuse, live-sibling position preservation,
  and token/full-vocabulary-logit agreement with fresh serial reference contexts.
- **15 numerical cases completed: eight passed, seven failed.** The seven failures
  are retained numerical observations, not passing correctness tests. See every
  exact command and both verification records in `numerical-final/manifest.json`.
- Serial f16/flash b2/P1024/S64 passes exactly for both arms, under both greedy
  generation and identical-input replay. At eight threads, both arms have the same
  independent-context failures: greedy token agreement 50.7812%, max difference
  1.464530; replay token agreement 85.9375%, max difference 0.845986. Their direct
  batched tree/naive comparisons still pass exactly.
- f32/no-flash replay fails even at one thread: 88.2812% token agreement and max
  difference 0.959662. The eight-thread control has the same values. f16/no-flash
  at eight threads also fails (max 1.330955). Neither precision nor serial execution
  is a universal numerical fix. Flash-off timings were therefore not collected.
- **Llama-3.2-1B Q4_K_M** passes both greedy and replay controls at b2/P128/S16/t1,
  f16/flash, including both verification scopes with zero differences. These are
  small second-model controls, not long-prefix or performance evidence.
- Current-binary batched b2 and b8/P1024/S200/t8/f16/flash checks pass with exact
  tokens/logits: **2,000 generated branch tokens**. The historical eight-shape,
  12,000-token checks remain separate Report 3 evidence, not new runs.

These controls establish equivalent batched outputs for the measured storage
comparison. They do not establish general numerical equivalence with independent
single-branch execution. Failure in the naive arm under the same control means
that discrepancy alone cannot be assigned to the radix tree. Its complete backend
root cause remains open.

## Repeated performance evidence

SmolLM2-135M-Instruct-Q4_K_M; prefix 1024; suffix 200; eight CPU threads; f16 KV;
flash on. **Five independently measured fresh-process trials per arm per workload**,
with a separate fresh-process warm-up for each: **20 measured trials total**.
MB below is decimal. Medians are computed per configuration from saved trials.

| Branches | Metric | Naive median | Tree median |
| --- | --- | ---: | ---: |
| 2 | Configured KV (MB) | 58.982 | 35.389 |
| 2 | Private process memory (MB) | 163.222 | 139.403 |
| 2 | Peak working set (MB) | 205.791 | 182.071 |
| 2 | Prefill (ms) | 18221.463 | 3015.321 |
| 2 | Decode-step p50 (ms) | 17.597 | 16.782 |
| 2 | Decode-step p99 (ms) | 95.096 | 1213.008 |
| 2 | Aggregate decode tokens/s | 83.736 | 21.850 |
| 8 | Configured KV (MB) | 235.930 | 64.881 |
| 8 | Private process memory (MB) | 344.482 | 171.901 |
| 8 | Peak working set (MB) | 386.597 | 214.733 |
| 8 | Prefill (ms) | 28817.512 | 2098.732 |
| 8 | Decode-step p50 (ms) | 43.935 | 39.095 |
| 8 | Decode-step p99 (ms) | 50.128 | 1475.684 |
| 8 | Aggregate decode tokens/s | 180.304 | 73.762 |

Configured KV is 40.0% lower at b2 and 72.5% lower at b8. Median private process
memory is 14.59% and 50.10% lower. Median prefill is shorter, but some prefill
observations vary substantially; use the saved quartiles/std and plots rather
than the medians alone. Configured capacity, Windows private commit and peak
working set are separate quantities; none measures unique physical KV pages.

Aggregate decode throughput is **73.91% lower at b2 and 59.09% lower at b8** in
this campaign. Median per-step times are lower, but much worse tails reduce total
throughput. The medians of per-trial p99 are about 1.21 and 1.48 seconds for tree.
These measurements do not support a decode speedup. All preplanned trials,
including slow runs, are retained; there are no significance claims.

Warm-up is in another process and does not warm the measured context's graph
state. Decode includes its first step and setup/scheduling tails, so this is not
an isolated steady-state kernel benchmark. The laptop was not thermally or
system-load isolated. The two plotted branch points are observations joined as
a visual guide, not a fitted scaling law. Do not pool this eight-thread campaign
with the older two-thread primary sample or older quantile calculations.

`measurements/matrix_coverage.csv` and `validation.json` show **20/80** original
primary trials covered and **60 unmeasured**. Branches 4/16, prefix 2048, repeated
thread sweeps and second-model performance are deferred. The high-priority subset
meets the report's commitment of three to five repeats per selected workload.

## Paired CPU profiles

`profiles-final/` contains **three runs per arm**, with the same binary, model,
b2/P1024/S200/t8/f16/flash workload and capacities as the measured b2 cases.
There are **21,823 instruction-pointer observations** across the six runs.
Initialization and module refresh succeeded in every replacement run. One run
recorded one failed thread suspension and one process-query failure; these missed
sampling opportunities remain visible in its metadata. Each child exited zero.

Every CPU-weighted decode address received a PDB or nearest-export label. However,
roughly **70-88% of the per-run weighted decode distribution uses DLL export
fallback**, not private PDB symbols. Frequent labels include ZwDelayExecution and
OpenMP runtime exports; resolved compute symbols include tinyBLAS quantized GEMM,
f16 vector dot/MAD and quantized vector dot routines.

This does **not** prove CPU time was spent inside a wait or identify the cause of
the measured tails. The delta describes preceding work, while the sampled pointer
is observed afterward, and suspension itself perturbs thread scheduling. No
on-CPU stacks or hardware counters were collected. See
[profiling method and limits](../../docs/PROFILING_METHOD.md). Scheduling/wait-policy
ablation or ETW on-CPU stacks would be a follow-up hypothesis test, not a result
of this collection. Instrumented timing never enters `raw_trials.csv`.

## Reproducibility and evidence index

- `ACCEPTANCE_AUDIT.json`: executable/model identity checks, source snapshot hashes,
  positive controls, exact coverage and per-profile naming/failure details.
- `validation/`: CTest summaries and detailed real test output, clean mock build
  logs, and the pre-fix regression failures.
- `numerical-final/`: 15 diagnostic records, raw stdout/stderr, commands and input
  source snapshots. Failed controls are explicitly retained.
- `measurements/`: 20 raw rows, source/model/binary environment identity, capacity
  discovery logs, independent warm-up/measurement logs, summaries and two figures.
- `profiles-final/`: six final raw collections, metadata, complete symbol CSVs,
  phase CPU timelines, summary tables and the flat-profile figure.
- `metric_comparisons.csv`: direct same-configuration median comparisons.
- `report-baseline.json`: digest and commitments extracted from the supplied PDF;
  the PDF and model binaries are not copied into Git.
- `numerical/` and `profiles/`: explicitly aborted/superseded collections. Their
  ABORTED notes explain exclusion; never pool them with final evidence.

Experiments record base commit `3dfab9c` plus their actual dirty source snapshots
and patches. The base hash alone does not identify the implementation. Compiled
benchmark inputs stayed fixed throughout the 20 timing runs and final profiles;
the executable SHA256 is `1345ce40e9dba705af5f888fc6184559c5c7756b18a4d64f77e4f7f2454391c3`.
Postprocessing was improved after collection; each final analysis_provenance.json
identifies the analyzer and exact raw inputs. Source snapshots are preserved byte
for byte, and may intentionally contain earlier postprocessing/auditor versions.

## Rerun entry points

```powershell
ctest --test-dir build -C Release --output-on-failure
ctest --test-dir build/wrapup-mock -C Release --output-on-failure
python scripts/run_wrapup_validation.py validation --output results/wrapup/reproduce-numerical
.\scripts\run_benchmarks_report3.ps1 -Experiment Matrix -OutputDir results/wrapup/reproduce-measurements -Branches 2,8 -Prefixes 1024 -Trials 5 -Threads 8 -KVType f16 -FlashAttention 1
python scripts/analyze_report3.py results/wrapup/reproduce-measurements
python scripts/run_wrapup_validation.py profiles --output results/wrapup/reproduce-profiles
```

Run the modes sequentially; use unused output directories. The acceptance auditor
checks the canonical saved `results/wrapup/` dataset:
`python scripts/audit_wrapup.py`. It verifies actual observations and fails if its
required positive controls or counts are incomplete. Build prerequisites and
optional IE_TEST_GGUF registration are in the README. Rebuilding or committing
changes campaign identity; preserve old data and use a new output directory.

## What is closed and what remains

The bounded CPU prototype now has regression/lifecycle tests, two-model controls,
a repeated memory/performance evaluation, matched named-address profiles and an
auditable demonstration package. That is the completed wrap-up work block.

General multithreaded/attention numerical invariance remains open. The current
raw-node API also does not close arbitrary scheduling, concurrent reclamation,
or ownership of multiple physical holders for identical complete token paths.
The repeated reuse test deliberately keeps a live anchor and divergent suffixes;
it does not prove all-branches-evicted prefix reuse. Adapter cell counters are
bookkeeping, not unique physical allocation measurements. These are explicit
research-prototype limits, not production-cache completion claims.

A fully general Pillar A implementation would need those ownership/API boundaries
resolved and tested. Pillars B/C, the omitted matrix cells and causal decode
optimization remain deferred. For the next progress report, claim a tested,
reproducible memory-sharing prototype with measured tradeoffs; do not claim the
entire original three-pillar system is finished or that every numerical issue is
resolved.
