# EXP-V2-0010: DeltaNet Transposed Wave Acceleration & Defeating mx-llama.cpp in Decode

## Question
Can we close the remaining 5.8 ms decode gap against `mx-llama.cpp` (38.9 ms/token, 25.7 tok/s) and achieve a clean, qualified end-to-end victory on AMD Instinct MI50 (gfx906, Wave64) by diagnosing and accelerating the 48 GDN DeltaNet recurrent state update stages within `PrefillV2Model`?

## Baseline & Competitor Context
- Hardware: 1 x AMD Instinct MI50 32GB (gfx906:sramecc+:xnack-, 60 CUs, Wave64, 1606 MHz SCLK, 1000 MHz MCLK, 225W)
- Model: `Qwen3.8-27B-Q4_K_M.gguf` (64 layers, 48 GDN SSM + 16 GQA attention)
- External Opponent (`mx-llama.cpp`):
  - Decode Latency: **38.90 ms/step** (25.7 tok/s)
  - P64 + TG128 End-to-End: **5.80 s**
  - P512 + TG128 End-to-End: **7.28 s**
- Prior MIInfer Baseline (V2-0009):
  - P64 / TG128: 44.24 ms/step (22.6 tok/s)
  - P512 / TG128: 44.73 ms/step (22.4 tok/s)
  - P2048 / TG128: 45.95 ms/step (21.8 tok/s)

## Root-Cause Profiling & Attribution

Diagnostic profiling via lightweight GPU event markers in `PrefillV2RecurrentLayer::decode_profiled` revealed that the 48 recurrent layers accounted for **82.4%** of total decode execution time (765 μs/layer vs 457 μs/layer for attention).

Detailed per-layer breakdown of `PrefillV2RecurrentLayer`:
- RMSNorm + Beta/Alpha Prep: 32.5 μs
- QKV + Gate Projections (Wave GEMV): 102.1 μs
- Conv1D + Head L2-Norm: 41.2 μs
- **GDN State Update (`launch_qwen35_deltanet_state_update`)**: **195.6 μs** (25.6% of entire recurrent layer!)
- SSM Postprocess + SSM Out Projection: 114.2 μs
- Residual Add + RMSNorm: 18.3 μs
- FFN Gate/Up + SwiGLU: 144.4 μs
- FFN Down + Residual Add: 116.8 μs

The legacy GDN single-step kernel was executing uncoalesced strided row-major global memory loads/stores across the 128x128 head state matrix, stalling wave execution on HBM2 memory latency.

## GDN Kernel Bakeoff

We benchmarked 5 candidate single-step GDN implementations in `bench/gdn_kernel_bakeoff.cpp`:

| Kernel Candidate | Architecture / Optimization | Step Latency | Full Model (48 Layers) | Speedup vs Baseline |
|:---|:---|---:|---:|---:|
| **1. Baseline Row-Major** | Strided global memory accesses | 195.56 μs | 9.39 ms | 1.00x |
| **2. Transposed** | Column-coalesced memory mapping | 151.92 μs | 7.29 ms | 1.29x |
| **3. Transposed No-Decay-Store** | Register-cached decay vectors | 114.04 μs | 5.47 ms | 1.71x |
| **4. Transposed LDS Inputs** | Query/Key broadcast in LDS | 91.00 μs | 4.37 ms | 2.15x |
| **5. Transposed Row Waves** | **Wave64 cooperative row mapping** | **54.75 μs** | **2.63 ms** | **3.57x (WINNER)** |

The `transposed_row_waves` kernel achieves **3.57x speedup**, trimming **140.81 μs per recurrent layer** and recovering **6.76 ms/token** across the 48 SSM layers.

Crucially, the transposed state layout in memory matches the chunked GDN prefill kernel (`m12_gdn_chunk_kernel`), preserving 100% zero-copy state handoff with zero transpose overhead.

## End-to-End Performance Results

Full model benchmark across prompt regimes on 1 x MI50 32GB:

| Metric | mx-llama.cpp | V2-0008 | V2-0009 | V2-0010 (This Work) | Delta vs mx |
|:---|---:|---:|---:|---:|:---|
| **P64 Decode (ms/tok)** | 38.90 ms | 55.36 ms | 44.24 ms | **36.45 ms** (27.4 tok/s) | **-2.45 ms (+6.6% faster)** |
| **P512 Decode (ms/tok)** | 38.90 ms | 56.20 ms | 44.73 ms | **37.31 ms** (26.8 tok/s) | **-1.59 ms (+4.3% faster)** |
| **P2048 Decode (ms/tok)**| 38.90 ms | 57.39 ms | 45.95 ms | **38.51 ms** (26.0 tok/s) | **-0.39 ms (+1.0% faster)** |
| **P64 + TG128 Wall (s)** | 5.80 s | 7.70 s | 6.28 s | **5.25 s** | **-0.55 s (+9.5% faster)** |
| **P512 + TG128 Wall (s)**| 7.28 s | 9.50 s | 8.04 s | **7.05 s** | **-0.23 s (+3.2% faster)** |

### Context Scaling Invariance
- $P64 \text{ Decode}: 36.45\text{ ms/step}$
- $P512 \text{ Decode}: 37.31\text{ ms/step}$ ($+2.3\%$)
- $P2048 \text{ Decode}: 38.51\text{ ms/step}$ ($+5.6\%$)
- Ratio $P2048 / P64 = 1.0565\times \le 1.10\times$ (Flat context scaling preserved).

### VRAM Footprint
- Model & State Allocation: 29.74 GiB
- Free VRAM Headroom: **2.26 GiB** ($\ge 2.0\text{ GiB}$ gate satisfied)

## Correctness Qualification
1. **Zero-Copy State Hand-off**: Token trajectory across 16 steps matches baseline bit-identically (`[7676, 220, 15, 25, 220, 15, 220, 19, 271, 220, 15, 11, 220, 198, 7676, 11]`).
2. **Multi-Turn Determinism**: Turn 1 vs Turn 3 bit-identical matching across 16 generated tokens following state reset.
3. **Unit Tests**: 24/24 CTest regression tests passing (100%).

## Decision
**KEEP**. Competitive Gate (< 38.9 ms/token) successfully cleared across all context regimes (P64: 36.45 ms, P512: 37.31 ms, P2048: 38.51 ms). MIInfer now outperforms `mx-llama.cpp` across the entire prompt + generation matrix on AMD Instinct MI50.
