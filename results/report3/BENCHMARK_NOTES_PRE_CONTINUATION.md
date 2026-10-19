# Benchmark Notes & Profiling Methodology — Senior Seminar Progress Report 3

## 1. Experimental Overview

This benchmark campaign investigates shared-prefix KV-cache management in LLM inference engines, evaluating the trade-offs between memory footprint, prefill execution, and decode step latency during branching (agentic tree-search) workloads.

### Primary Objectives
- **Milestone A (Correctness)**: Rigorously verify that shared-prefix tree execution produces deterministic, numerically equivalent tokens and logits compared to independent single-sequence execution.
- **Milestone B (Reproducibility)**: Structure fresh-process trial execution, isolate context capacity discovery from timed measurements, and collect multi-trial distributions (p50, p90, p99, std, IQR).
- **Milestone C (Decode Bottleneck Analysis)**: Systematically evaluate why KV sharing achieves massive memory reduction (up to ~80%) and prefill speedups (2x-5x) without a proportional reduction in per-step decode latency.

---

## 2. Hardware and Platform Environment

Collected from system environment probes (`environment.json`):
- **CPU**: AMD Ryzen 5 5500U with Radeon Graphics (6 Physical Cores, 12 Logical Processors, Zen 2 microarchitecture, 8 MB L3 Cache)
- **Host Memory**: 11.33 GB DDR4 (Dual-channel)
- **OS**: Microsoft Windows 11 Home (x64)
- **Compiler**: Microsoft Visual Studio 2022 / MSVC 19.37.32822.0 (C++20, flags `/W4 /WX /permissive-`)
- **llama.cpp Pinned Revision**: `434ddbbc0e30522e897670681e503b797c12b7c1` (v0.4.0-75-g434ddbbc0)
- **Primary Model**: `SmolLM2-135M-Instruct-Q4_K_M.gguf`
  - Layers ($L$): 30
  - Query Heads ($H$): 9
  - KV Heads ($H_{kv}$): 3 (Grouped Query Attention, GQA ratio 3:1)
  - Embedding Dimension ($D$): 576
  - Head Dimension ($D_{head}$): 64
  - Vocabulary Size ($V$): 49,152
  - Per-token KV cache cell footprint ($\kappa$):
    $$\kappa = 2 \times L \times H_{kv} \times D_{head} \times 2\text{ bytes} = 2 \times 30 \times 3 \times 64 \times 2 = 23,040\text{ bytes (22.5 KiB)}$$

---

## 3. Mathematical Foundations & Metrics Separation

Crucially, this study distinguishes three distinct memory concepts:

1. **Theoretical KV-Memory Model ($M$)**:
   For branch factor $b$, prefix length $P$, suffix length $S$, and cell size $\kappa$:
   $$M_{\text{naive}} = b \times (P + S) \times \kappa$$
   $$M_{\text{shared}} = (P + b \times S) \times \kappa$$
   $$\rho_{\text{theo}} = \frac{M_{\text{naive}}}{M_{\text{shared}}} = \frac{b(P + S)}{P + bS}$$
   Theoretical Peak Reduction:
   $$R_{\text{theo}} = 1 - \frac{1}{\rho_{\text{theo}}} = \frac{(b - 1)P}{b(P + S)}$$
   For $b=8, P=2048, S=200$:
   $$M_{\text{naive}} = 8 \times 2248 \times 23,040 = 414.36\text{ MB}$$
   $$M_{\text{shared}} = (2048 + 8 \times 200) \times 23,040 = 84.05\text{ MB}$$
   $$\rho_{\text{theo}} \approx 4.93\times \quad (\sim 79.7\%\text{ reduction})$$

2. **Configured Context Capacity ($n_{\text{ctx}}$ and $M_{\text{alloc}}$)**:
   The physical context slot capacity configured into the engine's memory pool:
   $$M_{\text{alloc}} = n_{\text{ctx}} \times \kappa$$
   Due to 256-cell allocation chunking / alignment padding, $n_{\text{ctx}}$ slightly exceeds raw token counts.

3. **Measured Process Memory**:
   Empirical memory reported by Windows PSAPI (`GetProcessMemoryInfo`):
   - `private_bytes`: Committed private bytes allocated by the process heap and runtime buffers.
   - `peak_ws_bytes`: Peak working set (resident physical memory pages in RAM).
   Measured working set includes static model weights (~100 MB for SmolLM2), runtime compute graphs (~25 MB), and KV cache buffers.

---

## 4. Correctness Verification Methodology (Milestone A)

The `--verify` CLI procedure validates the functional correctness of prefix sharing:
1. Executes the shared-prefix tree inference arm for a requested configuration $(b, P, S)$.
2. Independently spins up an isolated, single-sequence llama.cpp context for each branch $0 \le i < b$.
3. Feeds identical prefix tokens, identical initial suffix tokens, and evaluates greedy decoding for $S$ steps.
4. Extracts final logits via `llama_get_logits_ith` across the full vocabulary.
5. Computes:
   - **Token Match Rate**: Fraction of identical tokens generated across all branches and steps:
     $$\text{Match Rate} = \frac{\sum_{i=1}^b \sum_{s=1}^S \mathbb{I}[t_{i,s}^{\text{tree}} = t_{i,s}^{\text{indep}}]}{b \times S}$$
   - **Maximum Absolute Logit Difference**:
     $$\Delta_{\max} = \max_{i, v} |L_{i,v}^{\text{tree}} - L_{i,v}^{\text{indep}}|$$
   - **Mean Absolute Logit Difference**:
     $$\Delta_{\text{mean}} = \frac{1}{b \cdot V} \sum_{i=1}^b \sum_{v=1}^V |L_{i,v}^{\text{tree}} - L_{i,v}^{\text{indep}}|$$

### Tolerance Criteria
- Token match rate must be **$1.0$ ($100.0\%$)** under deterministic greedy decoding.
- Maximum absolute logit difference must be **$\le 1.0 \times 10^{-3}$**.
- Any mismatch terminates with process exit code `2`.

---

## 5. Decode Performance Bottleneck Analysis (Milestone C)

### Research Question
Why does prefix sharing dramatically reduce memory allocation (~80%) and prefill compute without yielding a proportional reduction in decode latency?

### Hypotheses & Experimental Findings

#### 1. Memory Bandwidth Bound vs Compute Bound
- Measured Memory Read Bandwidth (`ie_bandwidth_bench`):
  - 1 Thread: 13.25 GB/s
  - 2 Threads: 23.25 GB/s
  - 4 Threads: 23.14 GB/s
  - 8 Threads: 22.42 GB/s
- During a single decode step with $b=8$, the model reads:
  - Model weights: 105 MB (quantized Q4_K_M weights transferred per decode step)
  - KV Cache read traffic:
    In naive execution: $8 \times 2248 \text{ tokens} \times 23,040\text{ bytes/token} \approx 414\text{ MB}$ (if all layers read). But with GQA ($H_{kv}=3$), key-value heads are shared.
    In shared-prefix tree execution: although physical storage is unified, attention computation for each branch query still attends over the prefix length $P$. In standard multi-sequence attention without prefix-aware fused kernels, each query head computes attention over its active sequence positions, requiring attention reads proportional to $b \times (P + \text{step})$.
- **Key finding**: At batch size $b \ge 4$, execution transitions from memory-bandwidth latency bounded by weight reading to compute-bound GEMM/GEMV and attention score dot-products.

#### 2. Attention Kernel Fusions (Flash Attention)
- Testing `--flash-attn on` vs `--flash-attn off` revealed:
  - Enabled flash attention fuses attention graph nodes from 1,086 down to 936 nodes.
  - Decode step latency improved from ~11.89 ms down to ~10.10 ms (~18% throughput improvement).
  - However, the shared prefix still requires attending across $P$ positions for all $b$ queries unless cross-attention is parallelized across queries.

#### 3. CPU Thread Scaling (Zen 2 Architecture)
- Thread scaling from 1 to 4 threads shows strong throughput scaling (~2.5x speedup).
- Scaling beyond 4 to 8 threads exhibits diminishing returns due to memory channel saturation (dual-channel DDR4 reaches maximum ~23.2 GB/s at 2-4 threads) and SMT (hyperthreading) resource contention on 6 physical cores.

---

## 6. Exact Reproduction Commands

### Build Targets
```powershell
cmake -S . -B build -DIE_WITH_LLAMA=ON -DBUILD_TESTING=ON
cmake --build build --config Release --target kv_tree_tests memory_bench ie_bandwidth_bench ie_tree_bench
```

### Run CTest Suite (Including All Tree Edge Tests)
```powershell
ctest --test-dir build -C Release --output-on-failure
```

### Run Memory Bandwidth Calibration
```powershell
.\build\Release\ie_bandwidth_bench.exe --size-mb 256 --sweep --trials 5
```

### Run Real-Model Correctness Verification
```powershell
.\build\Release\ie_tree_bench.exe --model models\SmolLM2-135M-Instruct-Q4_K_M.gguf --arm tree --b 8 --prefix 2048 --suffix 64 --verify --single
```

### Run Benchmark Suite and Analysis
```powershell
.\scripts\run_benchmarks_report3.ps1
py scripts\analyze_report3.py
```

