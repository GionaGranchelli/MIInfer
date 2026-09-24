# V2-0002 — Prefill V2: Mx Compact Projection Backend & FAST_V1 Confrontation

## Hypothesis

Transplanting MIInfer's proven compact Mx MMQ projection backend (`launch_mx_q4k_repacked_mmq`, `launch_mx_q5k_repacked_mmq`, `launch_mx_q6k_repacked_mmq`, `launch_mx_q8_1_mmq_quantize`, `MxQ8_1MmqBlock`) into Prefill V2 will:
1. Reduce the total $N=512$ recurrent layer execution time from ~95.86 ms down to $\le 50$ ms ($\sim 2\times$ layer speedup).
2. Match or exceed the performance of the qualified production-shaped $B512$ `FAST_V1` path ($V2 / FAST\_V1 \approx 1.0\times$) while retaining V2's invariant semantic contract and clean multi-chunk execution ($C=64$).
3. Reduce persistent weight footprint by $\sim 60\text{--}75\text{ MiB}$ per recurrent layer ($\sim 3\text{--}4\text{ GiB}$ across 48 recurrent layers).
4. Maintain exact numerical invariance under sequence slicing ($512$ vs $256 + 256$) with cosine similarity $= 1.000000$ and max absolute error $= 0.0$.

## Motivation

In V2-0001, the clean-sheet Prefill V2 architecture proved structural validity and beat the sequential token oracle by $5.31\times$. However, V2-0001 reused the older M23 repacked MMQ projection backend, consuming ~79.5 ms in projection-bearing phases alone. This made V2-0001 substantially slower than the actual qualified production-shaped B512 V1 recurrent path (`FAST_V1` ~48 ms).

V2-0002 executes the primary optimization: migrating V2 to the compact Mx MMQ machinery without inheriting V1's brittle batch coupling.

## Environment

- **GPU**: 1 × AMD Instinct MI50 32 GB (gfx906, Vega 20, 60 CUs, Wave64)
- **ROCm / Compiler**: ROCm 6.2+ / hipcc (LLVM 18)
- **Model**: `Qwen3.8-27B-Q4_K_M.gguf` (`/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf`)
- **Layers Evaluated**:
  - **Signature A (24 layers)**: Layer 0 (Q6_K QKV, Q4_K Gate, Q5_K SSM-out, Q4_K FFN-g/u, Q6_K FFN-down)
  - **Signature B (24 layers)**: Layer 8 (Q4_K QKV, Q4_K Gate, Q5_K SSM-out, Q4_K FFN-g/u, Q4_K FFN-down)
  - **3-Layer Chain**: Layers 0 -> 1 -> 2 contiguous pipeline test

## Baseline vs Candidate

- **Oracle**: Canonical sequential V1 token path (`RecurrentLayer::run` token loop) — correctness reference.
- **FAST_V1**: Qualified production-shaped B512 Mx repacked prefill path (`RecurrentLayer::prefill_wide` with `MIINFER_PREFILL_WIDE_MX_REPACKED_MMQ=1`).
- **Candidate (V2-Mx)**: Clean-sheet Prefill V2 `PrefillV2RecurrentLayer` with resident compact Mx MMQ weights, monolithic 245.44 MiB workspace, and $C=64$ chunkwise GDN recurrence.

## Memory Footprint & Weight Compression

| Metric | V2-0001 (M23 MMQ) | V2-0002 (Mx MMQ) | Delta |
| :--- | :--- | :--- | :--- |
| **Signature A Persistent Weights / Layer** | 307.03 MiB | **246.88 MiB** | **-60.15 MiB (-19.6%)** |
| **Signature B Persistent Weights / Layer** | 288.62 MiB | **212.07 MiB** | **-76.55 MiB (-26.5%)** |
| **Projected 48 Recurrent Layers Weights** | 14.30 GiB | **10.76 GiB** | **-3.54 GiB** |
| **Recurrent States (48 Layers)** | 151.50 MiB | **151.50 MiB** | 0 |
| **Shared Recurrent Workspace (Monolithic)** | 247.00 MiB | **245.44 MiB** | -1.56 MiB |
| **Total Recurrent Static VRAM Footprint** | 14.69 GiB | **11.14 GiB** | **-3.55 GiB** |

## Empirical Results

### 1. Mandatory $N=512$ Baseline Comparison Table

| Layer / Signature | Oracle (ms) | FAST_V1 (ms) | V2-0001 (ms) | V2-Mx (ms) | V2 / FAST_V1 | V2 vs V2-0001 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Layer 0 (Sig A: Q6/Q6)** | 4,412.21 | 48.56 | 95.86 | **48.65** | **1.00×** (0.998×) | **1.97×** |
| **Layer 8 (Sig B: Q4/Q4)** | 4,098.48 | 47.11 | 89.40 | **47.12** | **1.00×** (1.000×) | **1.90×** |

### 2. Multi-Length Scaling ($N=64, 128, 512$)

| Layer / Tokens | Oracle (ms) | FAST_V1 (ms) | V2-Mx (ms) | Speedup vs Oracle | Throughput | Normalized Time |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Layer 0 (N=64)** | 551.09 | N/A (unsupported) | **11.09** | **49.68×** | 5,769 tok/s | 0.173 ms/tok |
| **Layer 0 (N=128)** | 1,069.42 | N/A (unsupported) | **14.28** | **74.91×** | 8,966 tok/s | 0.112 ms/tok |
| **Layer 0 (N=512)** | 4,412.21 | 48.56 | **48.65** | **90.69×** | 10,524 tok/s | 0.095 ms/tok |
| **Layer 8 (N=64)** | 505.58 | N/A (unsupported) | **10.65** | **47.46×** | 6,008 tok/s | 0.166 ms/tok |
| **Layer 8 (N=128)** | 1,011.02 | N/A (unsupported) | **13.79** | **73.34×** | 9,285 tok/s | 0.108 ms/tok |
| **Layer 8 (N=512)** | 4,098.48 | 47.11 | **47.12** | **86.99×** | 10,867 tok/s | 0.092 ms/tok |

### 3. Execution Phase Attribution ($N=512$, Layer 0)

| Phase | V2-0001 (M23) | V2-0002 (Mx MMQ) | Delta / Speedup | Share of Layer |
| :--- | ---: | ---: | ---: | ---: |
| **Phase 1: Norm & Beta/Alpha Prep** | 1.21 ms | **1.05 ms** | +15% | 2.15% |
| **Phase 2: QKV & Gate Proj (Mx MMQ)** | 17.03 ms | **7.35 ms** | **2.32×** | 15.06% |
| **Phase 3: Conv1D & L2 Norm** | 0.89 ms | **0.73 ms** | +22% | 1.50% |
| **Phase 4: Chunkwise GDN Core ($8\times C64$)** | 14.28 ms | **14.26 ms** | 1.00× | 29.22% |
| **Phase 5: SSM Post & Out (Q5_K MMQ)** | 7.32 ms | **2.94 ms** | **2.49×** | 6.03% |
| **Phase 6: Residual & Post Norm** | 0.10 ms | **0.10 ms** | 1.00× | 0.20% |
| **Phase 7: FFN Gate/Up (Q4_K MMQ)** | 38.54 ms | **13.84 ms** | **2.78×** | 28.36% |
| **Phase 8: SwiGLU, Down (Q6_K) & Residual** | 16.59 ms | **8.53 ms** | **1.95×** | 17.48% |
| **Total Layer Time** | **95.86 ms** | **48.65 ms** | **1.97×** | **100.0%** |

The projection-bearing phases dropped from **79.48 ms down to 32.66 ms** ($2.43\times$ faster projections).

### 4. Three-Layer Chain Semantic & Contiguous Timing ($L0 \rightarrow L1 \rightarrow L2$)

- **Numerical Verification vs Oracle**:
  - $N=64$: Final L2 Output Cosine = **0.999104**, RelRMS = 0.042397.
  - $N=128$: Final L2 Output Cosine = **0.999269**, RelRMS = 0.038424.
  - $N=512$: Final L2 Output Cosine = **0.999549**, RelRMS = 0.030282.
- **Contiguous N=512 Timed Interval**:
  - `FAST_V1 3-Layer Chain`: **145.90 ms** (min=145.62, max=146.19)
  - `Prefill V2 3-Layer Chain`: **145.99 ms** (min=145.71, max=153.35)
  - **Speedup vs FAST_V1**: **1.00× (0.999×)**

### 5. Stateful Split-Call Equivalence Test ($512$ vs $256 + 256$)

- Output Cosine Similarity: **1.000000** (MaxErr: 0.000000, RelRMS: 0.000000)
- Layer 0 State Cosine: **1.000000** (MaxErr: 0.000000)
- Layer 1 State Cosine: **1.000000** (MaxErr: 0.000000)
- Layer 2 State Cosine: **1.000000** (MaxErr: 0.000000)
- Layer 0 Conv History Cosine: **1.000000** (MaxErr: 0.000000)

Exact bitwise equivalence is preserved across multi-step chunked execution.

## Economic & Full-Model P512 Budget

- **Projected 48 Recurrent Layers Execution Time**:
  $$24 \times 48.653\text{ ms (Sig A)} + 24 \times 47.116\text{ ms (Sig B)} = \mathbf{2,298.46\text{ ms } (2.298\text{ s})}$$
- **External Competitor Benchmark**: `mx-llama.cpp` $P512 \approx 2,310\text{ ms } (2.31\text{ s})$.
- **Economic Budget Analysis**:
  The 48 recurrent layers occupy **2.298 s**. To beat `mx-llama.cpp` full-model $P512$, the remaining 16 GQA attention layers must execute within the target budget.

## Outcome Gate

**`V2_RECURRENT_PERFORMANCE_QUALIFIED`**

1. **Competitiveness**: V2-Mx matches the fastest qualified V1 recurrent path at $1.00\times$ speedup (48.65 ms vs 48.56 ms for L0; 145.99 ms vs 145.90 ms for 3-layer chain).
2. **Sub-512 Superiority**: V2-Mx cleanly executes $N=64$ (11.09 ms, 5,769 tok/s) and $N=128$ (14.28 ms, 8,966 tok/s), which V1 could not do without falling back to slow token loops.
3. **Memory Economy**: Saved 3.54 GiB VRAM across recurrent weights, enabling the full 48-layer recurrent model state + weights + workspace to reside comfortably in **11.14 GiB / 32 GiB**.
4. **Semantic Decoupling**: Exact invariant semantics ($512 \equiv 256 + 256$).

## Decision

**KEEP**. Prefill V2 recurrent execution is structurally qualified and fully performance-competitive.
