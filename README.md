# Inference Engine MVP

A standalone C++20 prototype for measuring KV-cache memory savings when agent branches share a prefix. It uses a mock allocator, so it does not load a model or depend on llama.cpp at runtime.

## Current MVP

- `KvRadixTree` stores shared token edges and dummy KV page IDs.
- Atomic node reference counts model branches retaining a shared prefix.
- `fork()` retains the existing path without copying its page IDs.
- `MockKvAllocator` accounts for KV bytes at `32 KiB` per token.
- The benchmark compares eight independent branches with one shared `2048`-token prefix and eight `200`-token suffixes.

This is a memory-accounting prototype. Structural updates are protected by a mutex; the implementation does not yet include EBR reclamation, eviction, slot multiplexing, real attention, or llama.cpp integration.

## Requirements

- Windows with Visual Studio 2022 and CMake 3.20 or newer
- A C++20-capable compiler

## Build and run

From the repository root in PowerShell:

```powershell
cmake -S . -B build
cmake --build build --config Release
.\build\Release\memory_bench.exe
```

Expected result for the built-in workload:

```text
naive KV allocation: 562.00 MiB
prefix-paged KV allocation: 114.00 MiB
peak allocation reduction: 79.72%
shared prefix ref_count: 8
```

The `build/` directory is ignored by Git. To try another workload, edit the constants near the top of `bench/main_bench.cpp`: branch count, prefix length, suffix length, or bytes per token.

## Layout

```text
core/kv_tree.hpp       Radix-tree interface and node definition
core/kv_tree.cpp       Tree ownership, append, fork, matching, and traversal
bench/memory_mock.hpp  KV byte-accounting allocator
bench/main_bench.cpp  Fixed workload and benchmark entry point
```

The memory comparison is:

```text
M_naive  = b * (P + S) * K
M_shared = (P + b * S) * K
```
