# Report 3 checkpoint: ready for a progress report

Work stops here at the user's requested checkpoint. This is a technical handoff,
not the academic report. No expanded performance campaign is running.

## What we achieved

- Fixed a real radix-tree split bug that could discard sibling sequences. The
  strengthened randomized check exposed it; the regression suite now has 18 cases.
- Built reproducible inference benchmarks with isolated capacity discovery,
  separate warm-up/measured processes, retained failures, source snapshots and
  model/executable hashes. Added thread, attention and KV precision controls.
- Completed all eight long **batched naive versus tree** correctness configurations:
  branches 2, 4, 8, 16; prefix 1024 or 2048; suffix 200; eight CPU threads;
  flash attention on; f16 KV; SmolLM2-135M-Instruct-Q4_K_M.
  All 12,000 generated branch tokens matched, and the recorded maximum and mean
  logit differences were zero in every configuration. This is one check per
  configuration, not five independent performance trials.
- Real-enabled CTest passed 4/4 targets, including two actual-model regressions;
  the mock build passed 2/2 targets. The model regressions cover f32/no-flash
  b2/P1024/S8/t2 and f16/flash b2/P1024/S64/t1. Test logs are retained here.
- Established a useful positive control: at one thread, f16/flash with
  b2/P1024/S64 matches independent-context execution exactly, as well as matching
  the batched baseline. Multithreaded independent-context mismatches remain.
- Added an optimized-PDB build option and a Windows child-process CPU sampler.
  One pilot collected 412 instruction-pointer samples with no recorded API
  failures. This establishes working profiling tooling, not a bottleneck result.

## Performance results available to the writer

Use the **previous committed** `../validated/` dataset and its methodology in
`../BENCHMARK_NOTES.md`. It contains 10 measured inference trials, 20 bandwidth
passes and 10 plots. Four inference trials form the primary comparison
(two independent trials per arm, b2/P1024/S200/t2, flash on, f16).
The other six are a preliminary thread sweep with one trial per arm at t1/t4/t8.
Do not combine this dataset with verification timings or profiler timings.

For that small primary comparison, median configured KV allocation was 58.98 MB
for naive versus 35.39 MB for tree (40% lower). Private process memory was about
14.7% lower, and prefill about 28.7% shorter. Decode throughput was about 15.8%
lower for tree, with worse tail latency. State these as observed results for
this configuration and sample, not universal speedups or statistical significance.
The bandwidth calibration reached about 25 GB/s at multiple thread counts;
this alone does not establish whether inference is compute or bandwidth bound.

The new directories `campaign/` and `repeated/` have **zero measured performance
rows**. Their header-only raw CSVs are intentional. `campaign/` is a failed f32
correctness control; `repeated/` contains the eight passing batched checks.
Neither directory establishes 80-trial benchmark coverage.

## Numerical correctness: what the evidence means

Two verification scopes are deliberately distinct. `--verify-batched` compares
batched naive execution directly against shared-tree execution. `--verify`
also compares against separate independent branch execution and keeps the strict
100% token match / maximum logit difference <= 1e-3 requirement. Passing the
first scope does not imply the second passed.

The eight long checks establish exact agreement between the two batched arms
under the tested f16/flash settings. They do not prove correctness for every
model, precision, backend, scheduling pattern or arbitrary branch input.
The synthetic workload is deterministic greedy decoding without EOS stopping.

At eight threads, b2/P1024/S8/f16/flash independent-context verification failed:
56.25% token match, max meaningful logit difference 0.859098. Disabling CPU fusion
or IQ-panel paths individually did not change that result. At one thread,
b2/P1024/S64/f16/flash passed exactly. An f32/no-flash eight-step test passed,
but its 64-step extension failed the numerical tolerance (max 1.303208) despite
100% token agreement; even its batched comparison failed. Precision alone is
therefore not a general fix. The failed records remain visible.

The pinned backend has distinct single-query split-KV and multi-query attention
paths and casts f32 K/V to f16 for CPU flash attention. These source observations
and controls suggest sensitivity to execution geometry and numerical reduction.
They do not establish a complete root cause or justify increasing tolerance.
No vendored backend code or submodule pin was changed.

## Evidence map and provenance

- `repeated/batched_correctness.csv`: eight long exact-parity checks.
- `repeated/environment.json`, `source.patch`, `source_snapshot/`, `logs/`:
  command settings, pinned dependency, SHA256 identities and raw records.
- `diagnostics/serial_f16_control.json` and corresponding logs: long serial pass.
- `diagnostics/initial_ablation.json`: multithreaded failure and two negative controls.
- `diagnostics/f32_control.json`: preliminary short precision controls.
- `campaign/correctness.csv`, `batched_correctness.csv` and logs: failed long f32 control.
- `ctest-real.log`, `ctest-mock.log`, `CHECKPOINT_AUDIT.json`: saved validation results.
- `profile-pilot/`: one instrumented f32/no-flash tree run, b2/P1024/S16/t8.
  Flat instruction pointers weighted by approximate per-thread CPU-time deltas;
  no call stacks, hardware counters or actual DRAM traffic. A substantial fraction
  is unresolved (roughly half the CPU weight). This is neither a paired arm
  comparison nor evidence for the primary f16/flash bottleneck. Profiling perturbs
  execution, so its latency must not enter benchmark statistics.

Experiments started from base commit `6356289` with uncommitted implementation
changes. Campaign source snapshots, hashes and patches identify their actual
inputs; the base commit by itself does not. The final eight-check executable
and all 21 source snapshot hashes were checked against the current workspace.
Preliminary diagnostics and the pilot were captured with earlier binaries;
use their own records and do not assign them the final executable hash.
New latency quantiles use linear interpolation (type 7); the older dataset used
lower order statistics. Keep the datasets separate. The metadata-only
`--attention-shapes` callback currently covers non-flash kq matrix multiplies;
no primary flash-attention shape study has been completed.

## Deferred work after the report

1. Run a larger independent performance matrix (target 80 trials), including
   repeated thread comparisons; report omissions explicitly if using a subset.
2. Investigate remaining independent-context numerical differences without
   relaxing the correctness threshold; extend model and divergent-input coverage.
3. Collect matched naive/tree CPU profiles with better symbol coverage before
   identifying a dominant decode kernel or claiming a bottleneck.
4. Complete a controlled attention ablation only where numerical validation
   supports comparison. Add second-model performance evidence if time permits.
5. Improve capacity discovery to avoid repeated requests that round to the same
   per-stream capacity. This optimization has been identified, not implemented.

## Reproduction entry points

```powershell
ctest --test-dir build -C Release --output-on-failure
ctest --test-dir build/report3_mock -C Release --output-on-failure
.\scripts\run_benchmarks_report3.ps1 -Experiment Parity -OutputDir results/report3/reproduce-parity -KVType f16 -FlashAttention 1 -AttentionModes 1 -Threads 8 -Branches 2,4,8,16 -Prefixes 1024,2048 -VerificationSuffix 200
```

Use an unused output directory after any source/binary change. Model acquisition
and build prerequisites are documented in the earlier technical handoff. The
optional model CTest targets require configuring `IE_TEST_GGUF`; Release PDBs
require `IE_PROFILE_SYMBOLS=ON`. Preserve the recorded failures when writing.

The report can now honestly tell this story: implemented shared-prefix storage,
fixed a structural bug, verified batched equivalence across the planned shapes,
measured meaningful memory savings in a small repeated comparison, and identified
numerical correctness and decode performance as remaining work.
