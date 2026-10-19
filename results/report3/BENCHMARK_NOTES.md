# Progress Report 3 technical handoff

This is implementation and experimental documentation, not an academic report.
The continuation began at commit `1a8b2e68e9083fdb14c0bbdcbb7075703862525e` with uncommitted checkpoint changes.
Previous results remain in this directory. New measurements are isolated in `validated/`;
the earlier failed verification is preserved in `continuation/`.
`BENCHMARK_NOTES_PRE_CONTINUATION.md` preserves the previous notes. Its compute-bound,
thread-scaling and flash-attention conclusions were not supported by the accompanying
CSV data and should not be cited as established findings.

## A. Implementation and audit

The existing repository already had compressed radix storage, atomic reference counts,
sequence copying, eviction, a mock backend, CPU llama.cpp inference, golden tests,
and a fresh-process single-arm comparison. Recent commits fixed reference ownership
across middle nodes and merging identical committed edges; those fixes were retained.
Pinned llama.cpp is `434ddbbc0e30522e897670681e503b797c12b7c1`.
Its local API exposes `n_threads`, `n_threads_batch`, `flash_attn_type`, `kv_unified`,
context/batch dimensions and f16 K/V defaults. No vendored code or pin was changed.

Files implemented by the checkpoint and completed in this continuation:

- `apps/ie_tree_bench.cpp`: configurable verification, thread/attention settings,
  explicit capacity, isolated discovery, warm-up control and machine-readable outputs.
  Fixed discovery returning the larger candidate after a smaller successful attempt.
  The search now descends on the 256-cell request grid until failure, handling
  backend rounding of non-unified contexts. This is a minimum observed on that grid,
  assuming capacity feasibility is monotonic; it is not a byte-granular allocator minimum.
  Fixed committed suffix labels to contain tokens actually fed to KV (initial input
  followed by predictions except the last), rather than the next-token predictions.
- `core/kv_tree.cpp`: a one-line fix preserves the sibling chain when splitting an
  existing edge. Strengthening the checkpoint's randomized assertion exposed lost
  sequences even though structural validation passed.
- `tests/kv_tree_tests.cpp`: 18 tree cases, including the focused sibling regression,
  fixed-seed insertion checks, forks, ownership, slot reuse, release order and eviction.
- `src/llm/llama_wrap.{hpp,cpp}`: forwards thread and flash-attention settings.
- `apps/ie_bandwidth_bench.cpp`, `CMakeLists.txt`: standalone C++20 read calibration,
  warm-up and repeated passes, an observable checksum and steady-clock timing.
- `scripts/run_benchmarks_report3.ps1`: Windows PowerShell-compatible resumable trials,
  capacity caching keyed by all runtime settings, separate warm-up/measured processes,
  retained stdout/stderr logs, failure exit handling, source/model/executable SHA256,
  source snapshots and immutable campaign identity. It rejects resuming after code,
  executable, model or baseline settings change; choose a new output directory.
- `scripts/analyze_report3.py`: rejects invalid/nonfinite measurements and duplicate
  trials; computes independent-trial median, IQR, min/max and sample standard deviation;
  separates controlled settings in figures and emits a missing-configuration audit.
  A modeled KV transfer estimate uses measured calibration bandwidth when available.

The checkpoint baseline ran before continuation edits: 2/2 CTest targets passed
(17 internal tree cases); mock golden output matched. The previous randomized test's
`matched-length OR non-null-node` check did not verify that inserted sequences remained
reachable. Replacing it with an exact matched-length check failed, then passed after
the sibling-chain fix. No expected outputs or tolerance were weakened.

## B. Validation

Primary build: MSVC 19.37.32822.0, Release, C++20, project `/W4 /WX /permissive-`.

```powershell
cmake -S . -B build -DIE_WITH_LLAMA=ON -DBUILD_TESTING=ON
cmake --build build --config Release --target kv_tree_tests memory_bench ie_bandwidth_bench ie_tree_bench
ctest --test-dir build -C Release --output-on-failure
.\build\Release\kv_tree_tests.exe
.\build\Release\memory_bench.exe
```

After the fix, both CTest targets pass and all 18 internal tree cases pass.
A separate clean mock-only Release build also passed 2/2 CTest targets:

```powershell
cmake -S . -B build/report3_mock -DIE_WITH_LLAMA=OFF -DBUILD_TESTING=ON
cmake --build build/report3_mock --config Release --target kv_tree_tests memory_bench ie_bandwidth_bench
ctest --test-dir build/report3_mock -C Release --output-on-failure
.\build\report3_mock\Release\memory_bench.exe
```

Invalid `--suffix 0` and `--flash-attn invalid` arguments were confirmed to return
exit code 1 before model loading. Campaign source and executable hashes match the
saved source snapshot. Analysis succeeded in a fresh Python process launched from
outside the repository and produced 10 PNG figures.
The mock benchmark prints 562.00 MiB naive and 114.00 MiB shared theoretical allocation,
79.72% reduction and prefix refcount 8. These are mock accounting results, not empirical
real-model memory reductions.

## C. Correctness and measured results

The strict criterion remains 100% greedy-token agreement and maximum absolute logit
difference <= `1e-3`; it was not increased. This is a fixed screening threshold for
CPU f16-KV inference, not an assertion of a universal backend error bound.
Verification runs separately after the timed run. Independent references use separate
contexts with the tested arm's capacity, KV layout, batch dimensions, thread count,
and attention setting, but only one active sequence in each context and no sequence
copy. Logits are compared at every step while branch input histories match, including
the first differing prediction. Subsequent different-history logits are excluded.
Free-running token agreement still includes every step.

A supplemental `CSV_BATCH_VERIFY` compares batched naive versus batched tree inference
in an additional untimed run. This does not replace the independent-context check.

Recorded on SmolLM2-135M-Instruct-Q4_K_M, 2 threads, flash attention enabled:

| b / prefix / suffix | Batched naive vs tree | Separate-context reference |
|---|---|---|
| 2 / 128 / 64 | 100% tokens; max/mean logit difference 0 | 100% tokens; max/mean difference 0; PASS |
| 2 / 1024 / 64 | 100% tokens; max/mean logit difference 0 | 26.5625% tokens; max 1.194773; mean 0.129118; FAIL (exit 2) |

With flash attention disabled at `2 / 1024 / 64`, both the batched comparison and
independent-context check have 100% token agreement but maximum logit difference
1.183932 and mean difference 0.153552; both FAIL the same numerical criterion.
These final-source measurements are in `correctness.csv`, `batched_correctness.csv`
and the verification logs. The optional disabled/enabled latency sweep was skipped
because disabled-mode batched numerical parity was not established.

At prefix 1024, the first divergent predictions occur at branch 0 step 3 and branch 1
step 1 (zero based). Shorter diagnostic runs reproduced exactly the same metrics and
first divergences for both naive and tree versus independent execution. With attention
disabled, a 16-step diagnostic had 100% token agreement but maximum logit difference
0.826392 (still FAIL). The diagnostic outputs are in the ignored build directory;
the final-source 64-step enabled results and full backend logs are preserved in
`validated/`. Do not cite different-history final-logit differences from older runs
as numerical error under identical inputs.

Actual primary subset: SmolLM2, b=2, P=1024, S=200, 2 threads, enabled flash attention,
two measured fresh processes per arm, each preceded by one separate warm-up process.
Statistics below are medians over those two trials. MB uses 1,000,000 bytes.

| Metric | Naive | Tree |
|---|---:|---:|
| Minimum observed 256-cell-grid context slots | 2560 | 1536 |
| Estimated configured KV capacity (MB) | 58.982 | 35.389 |
| Process private memory (MB) | 162.447 | 138.648 |
| Process peak working set (MB) | 205.058 | 181.152 |
| Prefill latency (ms) | 10054.596 | 7168.986 |
| Decode-step p50 (ms) | 28.985 | 28.465 |
| Decode-step p90 (ms) | 34.150 | 47.025 |
| Decode-step p99 (ms) | 38.397 | 120.853 |
| Aggregate decode tokens/s | 67.603 | 56.892 |

Configured KV capacity is 40.0% smaller; private memory is 14.65% smaller and working
set is 11.66% smaller. The theoretical live-KV model predicts 56.402 MB versus
32.809 MB (41.83% reduction), which is a different quantity. Prefill median is 28.70%
lower; throughput median is 15.84% lower. This small sample does not establish a
statistically significant improvement or regression. Raw trials, IQR and sample
standard deviations remain in `raw_trials.csv` and `summary.csv`.

C1 exploratory thread scaling uses the same b=2/P=1024/S=200 workload and enabled
attention. One trial per arm at 1, 4, 8 threads; two per arm at 2 threads:

| Threads | Naive step p50 (ms) | Tree step p50 (ms) | Naive tokens/s | Tree tokens/s |
|---:|---:|---:|---:|---:|
| 1 | 48.482 | 59.140 | 40.522 | 31.498 |
| 2 | 28.985 | 28.465 | 67.603 | 56.892 |
| 4 | 25.623 | 60.786 | 42.542 | 29.650 |
| 8 | 18.450 | 17.282 | 91.837 | 103.785 |

Eight threads produced the highest throughput in this sample. Four-thread tails
were large (naive p99 177.383 ms, tree p99 161.025 ms). Single-trial points and
uncontrolled machine conditions preclude declaring an optimal thread count or
attributing those stalls to scheduling, bandwidth or compute saturation.

C2 median sequential-read bandwidth over five passes: 14.7819, 25.0166, 25.6143,
25.1316 GB/s at 1, 2, 4, 8 threads. Individual printed pass values (two decimal
places) are in `bandwidth_trials.csv` and the full captured log; timestamped bandwidth
summary CSV retains four-decimal aggregate values emitted by the executable.
Native cache enumeration found two 4 MiB L3 caches; the 256 MiB buffer is much larger
than their combined capacity. For this workload the illustrative logical-KV read
model gives 51,816,960 bytes/step, or 2.0713 ms at the measured two-thread calibration
bandwidth. This is modeled traffic, not observed llama.cpp DRAM traffic.

Only 4 of 80 requested primary trials were measured. Fourteen of sixteen primary
arm/configuration combinations are absent; the two present combinations have two
of five target trials each. The six other performance trials are thread-scaling
explorations and do not fill those missing primary repetitions. See `coverage.json`
and `experiment_status.json`.

## D. Scientific interpretation

Direct batched comparison provides evidence that KV prefix sharing preserves this
workload's outputs. The independent-context mismatch also occurs without prefix
sharing, which narrows investigation to backend execution differences associated
with batch shape and attention/KV layout. The disabled-attention batched logit
mismatch additionally shows numerical dependence on execution configuration.
The exact responsible kernel has not been established; no claim
of complete correctness across all supported configurations is warranted.

The measured enabled-attention subset uses less configured KV capacity and has
lower median prefill latency, while decode p50 is similar and decode tails/throughput
are worse at two threads. These observations are compatible with several bottleneck
hypotheses; they do not identify one.

KV storage sharing reduces physical capacity and prefix recomputation. Each branch
still computes its own next-token distribution. Latency and a sequential read calibration
alone cannot establish compute saturation, CPU utilization, cache reuse or llama.cpp
memory traffic. Hardware counters or a kernel profiler would be needed for those claims.

`modeled_kv_transfer.csv`, when generated, assumes logical f16 K/V reads of
`b * (P + (S + 1)/2) * kappa` bytes per lockstep step on average, divided by measured
sequential-read bandwidth. It ignores weight traffic, cache hits, repeated query-head
reads, padding and compute. It is an illustrative transfer-time floor under this
model, not a measured latency bound for llama.cpp or a measurement of its DRAM traffic.

## E. Limitations and metric definitions

- The full 80-trial primary matrix is not complete. See the new CSVs and
  `validation.json` for measured and missing configurations; historical 15 single
  trials have different settings and are not pooled with the continuation.
- Opt-in verification retains per-step vocabulary logits, so its memory grows as
  `branches * suffix * vocabulary`. This allocation occurs in the untimed verification
  run, after the measured process-memory sample.
- Independent-context verification remains failing at the required longer prefix.
  The runner preserves failures and returns nonzero; a full `All` campaign stops
  before profiling on that failure. Explicit performance runs characterize batched
  naive/tree execution only; they do not establish single-sequence equivalence.
- CPU: Ryzen 5 5500U, 6 physical / 12 logical cores (native psutil query); current
  usable RAM 12,240,412,672 bytes. Windows version and exact environment are in JSON.
  Machine load, power/thermal conditions and CPU affinity were not controlled.
- `kv_bytes = n_ctx * kappa` estimates f16 allocated KV capacity, not process memory.
  The theoretical live-token formulas are `b*(P+S)*kappa` and `(P+b*S)*kappa`.
  Windows private bytes and peak working set include weights, graphs, output buffers,
  allocator/runtime overhead and KV. Separate-process warm-ups avoid including
  discovery/verification peaks in measured-process memory.
- Latency percentiles use the existing lower order statistic `floor(q*(n-1))`.
  With only two fork samples (including the original sequence's retain), the reported
  fork p99 is particularly coarse; it must not be interpreted as a calibrated
  sequence-copy tail latency. Naive fork zeros denote no fork operation.
- Throughput is decode-only: generated tokens divided by the sum of timed lockstep
  decode calls. It excludes prefill, forks and loop bookkeeping. Tail stalls can
  lower throughput even when step p50 improves.
- Step p50/p90/p99 are within-trial distributions. Summary median, IQR and sample
  standard deviation use process trials. Single-trial std is undefined (blank),
  not zero. Descriptive error bars are not confidence intervals.
- Bandwidth calibration reads a 256 MiB buffer, well above the CPU's LLC. Timing
  includes worker creation/join and checksum work; passes share one process/buffer.
  It neither measures llama.cpp traffic nor establishes its bottleneck.
- Optional Llama-3.2-1B is available locally but has not been measured in this
  continuation. No models were downloaded.

## F. Reproduction

Run from the repository root. Python analysis needs pandas, NumPy and Matplotlib.
Build using section B. Use a new output directory if source/binary hashes differ
from the saved campaign; snapshots preserve the source used for each campaign.

```powershell
# Small strict pass, followed by a larger strict failure that records diagnostics:
.\scripts\run_benchmarks_report3.ps1 -OutputDir results/report3/new-run -Experiment Correctness -Branches 2 -Prefixes 128 -AttentionModes 1 -Trials 2
.\scripts\run_benchmarks_report3.ps1 -OutputDir results/report3/new-run -Experiment Correctness -Branches 2 -Prefixes 1024 -AttentionModes 1 -Trials 2

# Representative primary subset; discovery and warm-up are not measured trials:
.\scripts\run_benchmarks_report3.ps1 -OutputDir results/report3/new-run -Experiment Matrix -Branches 2 -Prefixes 1024 -Trials 2 -MaxTrials 4
# Short profiling chunks, re-run to resume:
# Executed three times to collect 1-, 4-, and 8-thread pairs (2-thread rows already exist):
.\scripts\run_benchmarks_report3.ps1 -OutputDir results/report3/new-run -Experiment Threads -ProfileBranches 2 -ProfilePrefix 1024 -Trials 1 -MaxTrials 2
# Disabled-attention correctness check: expected strict failure, recorded separately:
.\scripts\run_benchmarks_report3.ps1 -OutputDir results/report3/new-run -Experiment Correctness -Branches 2 -Prefixes 1024 -AttentionModes 0 -Trials 1
# Optional latency sweep was NOT executed because the preceding numerical check fails:
# .\scripts\run_benchmarks_report3.ps1 -OutputDir results/report3/new-run -Experiment Attention -ProfileBranches 2 -ProfilePrefix 1024 -Trials 2 -MaxTrials 4
.\scripts\run_benchmarks_report3.ps1 -OutputDir results/report3/new-run -Experiment Bandwidth -Trials 5
python scripts/analyze_report3.py results/report3/new-run

# Full requested primary matrix, resumable in four-trial chunks:
.\scripts\run_benchmarks_report3.ps1 -OutputDir results/report3/full-matrix -Experiment Matrix -Trials 5 -MaxTrials 4
```

The full matrix command does not imply strict verification passed; review correctness
results before treating it as evidence of independent-sequence equivalence.

## G. Git history and publication

The implementation is committed as:

- `d6ba53d` ? Strengthen KV tree correctness and inference verification.
- `9451658` ? Add reproducible benchmark runner and CPU profiling tools.
- Results, plots, README and this handoff are included in the accompanying results commit.

The preceding committed suite contained 7 tree cases; the current suite contains 18.
The earlier 17-case checkpoint baseline already included uncommitted tests from the
previous agent, so it is a different comparison from the committed 7-case baseline.

Measurements were taken before these commits, starting from base HEAD
`1a8b2e68e9083fdb14c0bbdcbb7075703862525e` plus the recorded changes. Campaign source
and executable hashes were verified against the saved source snapshot. The manifests
retain their original experimental provenance; they have not been rewritten to imply
that the later commits were used during measurement.

Because the runner includes HEAD in campaign identity, additional measurements after
committing must use a new output directory. Preserve the existing campaign and compare
only configurations with matching runtime settings. Final push status is reported in
the session handoff after remote verification.
