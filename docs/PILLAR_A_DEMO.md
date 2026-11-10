# Pillar A demonstration

Run these from the repository root after the real-model build described in the
README. Keep the pinned backend and the recorded model files unchanged.

1. Run the dependency-free accounting demonstration:

   ```powershell
   .\build\Release\memory_bench.exe
   ```

   This is a mock allocation model, not measured GGUF RAM usage. The golden output
   shows 562 MiB naive versus 114 MiB logical shared allocation for its built-in
   eight-branch workload. CTest protects the exact output.

2. Demonstrate actual backend lifetime handling:

   ```powershell
   .\build\Release\ie_lifecycle_tests.exe models/SmolLM2-135M-Instruct-Q4_K_M.gguf
   ```

   Expect 12 `LIFECYCLE,...,PASSED` records. The test forks an internal prefix from
   a longer live descendant, checks that suffix state was excluded, generates a
   new suffix, compares with a fresh serial reference, releases/evicts that suffix,
   and reuses the slot. The other branch remains live throughout.

3. Demonstrate strict same-input numerical verification in a passing configuration:

   ```powershell
   .\build\Release\ie_tree_bench.exe --model models/SmolLM2-135M-Instruct-Q4_K_M.gguf --arm tree --b 2 --prefix 1024 --suffix 64 --threads 1 --flash-attn on --kv-type f16 --n-ctx 1536 --verify --teacher-forced
   ```

   Expect both `CSV_BATCH_VERIFY` and `CSV_VERIFY` to report PASSED. This is
   deterministic input replay; remove `--teacher-forced` to demonstrate the tested
   serial greedy configuration. These diagnostic timings do not enter the
   repeated benchmark dataset.

4. Show `results/wrapup/measurements/summary.csv`, its figures, and the coverage
   audit. Distinguish configured KV capacity, OS process memory and throughput.
   Show the failed numerical controls too: the prototype does not claim general
   thread/attention numerical invariance or a universal decode speedup.

The complete test entry point is `ctest --test-dir build -C Release --output-on-failure`.
The remaining two original project pillars are deferred. This demonstration
supports the bounded CPU research prototype described in PILLAR_A_ACCEPTANCE.md.
