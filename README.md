# Inference Engine

A C++20 prototype for measuring shared-prefix KV-cache behavior in agentic tree search. It supports a dependency-free mock path and an opt-in CPU llama.cpp path.

## Current MVP

- `KvRadixTree` stores token edges and leaf `SeqRange` holders.
- Atomic reference counts, split-on-insert, slot allocation, release, LRU idle-leaf eviction, and validation.
- `MockBackend` reproduces the original accounting benchmark.
- The optional llama.cpp wrapper supports CPU prefill, greedy lockstep decode, sequence copy/remove, and model KV metadata.
- `ie_tree_bench` runs one fresh-process real-model arm; `scripts/run_tree_bench.ps1` aggregates the sweep.

The current tree uses a mutex for structural operations. Epoch reclamation, real-model measurements, and profiling extras remain pending.

## Requirements

- Windows with Visual Studio 2022 and CMake 3.20 or newer
- A C++20-capable compiler
- The vendored llama.cpp submodule for real-model targets
- GGUF files at `models/SmolLM2-135M-Instruct-Q4_K_M.gguf` and `models/Llama-3.2-1B-Instruct-Q4_K_M.gguf` for smoke/benchmark runs

## Download models

Install the Hugging Face CLI and download the exact files into the ignored `models/` directory:

```powershell
py -m pip install -U huggingface_hub
hf download bartowski/SmolLM2-135M-Instruct-GGUF SmolLM2-135M-Instruct-Q4_K_M.gguf --local-dir models
hf auth login
hf download bartowski/Llama-3.2-1B-Instruct-GGUF Llama-3.2-1B-Instruct-Q4_K_M.gguf --local-dir models
```

The SmolLM2 repository is public. Llama 3.2 requires accepting Meta's license on the model page and authenticating the Hugging Face CLI. Keep the access token private.

## Build and run

From the repository root in PowerShell, the mock build is:

```powershell
cmake -S . -B build
cmake --build build --config Release --target memory_bench kv_tree_tests
.\build\Release\memory_bench.exe
ctest --test-dir build -C Release --output-on-failure
```

Expected result for the built-in workload:

```text
naive KV allocation: 562.00 MiB
prefix-paged KV allocation: 114.00 MiB
peak allocation reduction: 79.72%
shared prefix ref_count: 8
```

The `build/` directory is ignored by Git. The golden benchmark output is covered by CTest.

To build the real backend and smoke app:

```powershell
git submodule update --init --recursive
cmake -S . -B build -DIE_WITH_LLAMA=ON -DBUILD_TESTING=ON
cmake --build build --config Release --target ie_smoke ie_tree_bench
.\build\Release\ie_smoke.exe models\SmolLM2-135M-Instruct-Q4_K_M.gguf
```

Run one tree benchmark case:

```powershell
.\build\Release\ie_tree_bench.exe --model models\SmolLM2-135M-Instruct-Q4_K_M.gguf --arm tree --b 8 --prefix 2048 --suffix 200 --single
```

Run the full fresh-process sweep:

```powershell
.\scripts\run_tree_bench.ps1 -BuildDir build\Release
```

## Status

| Level | Path | Status |
| --- | --- | --- |
| L0 | Mock backend and radix tree | Implemented; golden output and CTest pass |
| L1 | llama.cpp backend and real-model benchmark | Compiled; runtime measurements require the GGUF files |

## Layout

```text
core/kv_backend.hpp   Backend interface and mock accounting backend
core/kv_tree.hpp      Radix-tree interface and node definition
core/kv_tree.cpp      Tree ownership, split, fork, release, eviction, and validation
src/llm/              llama.cpp wrapper and backend adapter
apps/                 Smoke and real-model tree benchmark entry points
tests/                Property, differential, split, slot, and golden tests
scripts/              Fresh-process benchmark sweep
```

The memory comparison is:

```text
M_naive  = b * (P + S) * K
M_shared = (P + b * S) * K
```
