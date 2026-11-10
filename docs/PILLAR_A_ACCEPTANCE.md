# Pillar A wrap-up scope (three-week work block)

The submitted October 19 Progress Report 3 is the baseline. The revised objective
is to finish one defensible prototype contribution: reference-counted shared-prefix
KV storage. Speculative tool scheduling (B) and vectorized grammar masking (C)
are deferred. Work recorded in this repository is actual implementation and
experiments, not a claim that three calendar weeks have already elapsed.

## Week 1: correctness and lifecycle closure

- Exercise internal-prefix re-forking after suffix generation, failed-copy recovery,
  explicit sequence-slot ownership, release, eviction, and repeated slot reuse.
- Preserve regressions failing before their fixes, and run mock and real-model CTest.
- Compare identical branch-specific input histories across serial/batched execution,
  thread counts, attention modes and KV precision, preserving the 1e-3 threshold.
- Add a second-model control and distinguish tested configurations from general
  numerical invariance. A backend mismatch is an open limitation, not a pass.

## Week 2: repeated measurements

- High-priority workloads: SmolLM2-135M Q4_K_M; b=2 and b=8; P=1024; S=200;
  eight threads; flash on; f16 KV. Five fresh-process trials per arm: 20 total.
- Keep capacity search, verification and profiling outside those timing trials.
- Record source, dependency, executable and model identities; retain raw logs.
- Fresh-process warm-ups warm system/model-file state, not the measured context's
  graph state. The 200 measured decode steps include the first decode step and
  any graph/setup or scheduling tails; do not call this steady-state throughput.
- Report median, IQR and sample spread, both arms' memory and decode throughput,
  and audit missing cells against the original 80-trial goal.
- Omitted branch factors/prefixes and second-model performance remain explicit.

## Week 3: reproducible evaluation and handoff

- Collect three paired naive/tree CPU profiles for the exact same measured
  b=2/P1024/S200/t8/f16/flash workload and optimized binary.
- Resolve instruction pointers while the child is alive. Report unresolved
  fractions, sampling perturbation and the absence of hardware counters/stacks.
- Generate figures and tables only from saved records. Keep profiles out of
  latency statistics. Do not infer DRAM saturation from streaming bandwidth.
- Publish a validation index, supported scope, known limitations, exact rerun
  commands, and a concise next-report evidence handoff.

## Acceptance boundary

This is a CPU, single-writer research prototype using the pinned llama.cpp
revision, synthetic deterministic token inputs, and explicit branch lifetimes.
Completion means that the defined support boundary is tested and reproducible;
it does not mean all models and numerical execution geometries are equivalent.

Required positive evidence:

1. All structural/regression tests and the golden mock benchmark pass.
2. The real-model lifecycle test passes 12 reuse cycles, verifies copied/evicted
   position bounds, preserves a live sibling, and checks tokens and logits against
   fresh serial references at the unchanged tolerance.
3. Long b=2/b=8 batched tree-versus-naive parity checks pass on the current binary.
4. Five independent performance trials exist for each of the four declared arm/
   workload cells. No verification/profile timing is substituted.
5. Matched profiles, reproducibility identities, omitted coverage and known
   numerical failures are retained. Bottleneck claims require resolved evidence.

Remaining research boundaries include multithreaded independent-context numerical
invariance, arbitrary scheduling/concurrency, identical complete-path physical
holder ownership, true KV physical-page accounting, and broader model coverage.
`cells_in_use()` in the adapter is bookkeeping, not a measurement of unique
physical pages; eviction reports removed logical edge tokens. Context destruction
ultimately releases backend storage. The benchmark reports configured capacity
and OS process memory separately. Raw `Node*` handles require external lifetime
coordination; the mutex does not establish a concurrent reclamation protocol.

If a required positive control fails, the handoff must state which criterion
remains open. Do not label a numerical failure a passing correctness result or
weaken thresholds to close the milestone.
