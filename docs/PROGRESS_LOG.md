# Progress Log

## Step 1: llama wrapper

Command:

```powershell
cmake -S . -B build-llama-clean -DIE_WITH_LLAMA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-llama-clean --config Release --target ie_smoke memory_bench
```

Result: llama.cpp, `ie_llm`, `ie_smoke`, and `memory_bench` compiled under MSVC. The requested model paths were absent, so model execution was not run.

## Step 2: backend abstraction

The build linked the vendored `llama` target and compiled `LlamaBackend`. The mock benchmark remained:

```text
naive KV allocation: 562.00 MiB
prefix-paged KV allocation: 114.00 MiB
peak allocation reduction: 79.72%
radix-tree nodes: 10
shared prefix ref_count: 8
```

## Step 3: tree lifecycle

Added slot allocation, split-on-insert, release, idle-leaf eviction, validation, exception-safe publication, and iterative teardown. The same golden output passed after the changes.

## Step 4: tests

Command:

```powershell
ctest --test-dir build-step4 -C Release --output-on-failure
```

Result:

```text
100% tests passed, 0 tests failed out of 2
```

The suite includes the 1,000-operation property test, differential matching, split edge cases, slot exhaustion, and exact mock output validation.

## Step 5: real-model benchmark

`ie_tree_bench` and `scripts/run_tree_bench.ps1` compile and the PowerShell script parses successfully. The real-model run is blocked until these files exist:

```text
models/SmolLM2-135M-Instruct-Q4_K_M.gguf
models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
```

No latency, RSS, KV-capacity, or correctness numbers are recorded until the models are available.