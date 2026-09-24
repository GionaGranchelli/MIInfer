# V2-0001 — Clean-Sheet Single-MI50 Prefill V2: Recurrent-Layer Vertical Slice

## Hypothesis

A clean-sheet Prefill V2 architecture designed exclusively for single-MI50 (gfx906, Wave64) with explicit recurrent state flow, resident MMQ-tiled weights, pre-allocated workspace, and native chunkwise GDN ($C=64$) will:
1. Preserve invariant model semantics across prompt lengths without physical batch coupling.
2. Deliver $> 1.5\times$ speedup over the canonical V1 token oracle on a complete real Qwen3.8-27B-Q4_K_M recurrent layer.
3. Monotonically reduce per-token execution time as prompt length increases from $N=64$ to $N=512$.

## Motivation

Legacy Prefill V1 tightly coupled prompt length, physical matrix batch width, causal chunk size, and workspace assumptions. This produced divergent semantic routes ($B512$, $B128$, $B64$, $B4$) with pathological slowdowns and numerical drift on partial tails. Prefill V2 decouples these concerns completely, using a fixed mathematical chunk $C=64$ for recurrence and explicit state contracts.

## Environment

- **GPU**: 1 × AMD Instinct MI50 32 GB (gfx906, Vega 20, 60 CUs, Wave64)
- **ROCm / Compiler**: ROCm 6.2+ / hipcc (LLVM 18)
- **Model**: `Qwen3.8-27B-Q4_K_M.gguf` (`/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf`)
- **Layer Selected**: Layer 0 (Representative Gated DeltaNet recurrent layer containing complete quantization mix: Q4_K projections, Q5_K SSM out, Q6_K FFN down, and F32 norms/conv/scales).

## Baseline vs Candidate

- **Baseline (Oracle)**: Frozen V1 canonical token-recurrent layer execution (`RecurrentLayer::prefill_token_loop` with exact sequential state evolution).
- **Candidate**: Prefill V2 `PrefillV2RecurrentLayer` with resident MMQ-tiled weights, unified 247 MiB workspace, and fused chunkwise GDN ($C=64$).

## Compiler Resource & Occupancy Analysis

| Kernel | Target | SGPRs | VGPRs | LDS (Bytes) | Scratch / Spill | Wave Occupancy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `m12_gdn_chunk_kernel` | gfx906 | 75 | 26 | 0 | 0 B | 9 waves / SIMD (max 10) |
| `gemv_mmq_q4k_f32_affine_repack` | gfx906 | 64 | 40 | 0 | 0 B | 6 waves / SIMD |
| `gemv_mmq_q5k_f32_affine_repack` | gfx906 | 64 | 42 | 0 | 0 B | 6 waves / SIMD |
| `gemv_mmq_q6k_f32_affine_repack` | gfx906 | 64 | 44 | 0 | 0 B | 6 waves / SIMD |

Zero register spills or scratch allocations exist across the entire layer execution path.

## Memory Footprint

- **Persistent Layer Weights**: 307.03 MiB (resident device memory).
- **Transient Layer Workspace**: 247.00 MiB (allocated once for $N \le 512$, 0 B hot-path allocation).
- **Recurrent State per Layer**: 3.15 MiB matrix state ($[48, 128, 128]$ F32) + 163.84 KB conv history ($[4, 10240]$ F32).

## Empirical Results

### 1. Numerical Equivalence Against Canonical V1 Oracle

| Prompt Length | GDN Chunks | Output Cosine Sim | Output Rel RMS | State Cosine Sim | Conv Cosine Sim | Finite Check |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **N = 64** | $1 \times C64$ | **0.998167** | 0.062214 | **0.999812** | **1.000000** | PASS (0 NaNs/Infs) |
| **N = 128** | $2 \times C64$ | **0.998413** | 0.057340 | **0.999794** | **1.000000** | PASS (0 NaNs/Infs) |
| **N = 512** | $8 \times C64$ | **0.999543** | 0.030612 | **0.999881** | **1.000000** | PASS (0 NaNs/Infs) |

All outputs and states match canonical token recurrence within standard Q4_K_M precision envelopes with zero divergence over extended sequence lengths.

### 2. Performance Bakeoff (Single Recurrent Layer on MI50)

| Prompt Length ($N$) | V1 Token Oracle (ms) | Prefill V2 (ms) | Speedup | Throughput (tok/s) | Normalized (ms/tok) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **64** | 84.09 ms | **20.63 ms** | **4.08×** | 3,103 tok/s | 0.322 ms/tok |
| **128** | 126.73 ms | **28.59 ms** | **4.43×** | 4,478 tok/s | 0.223 ms/tok |
| **512** | 508.84 ms | **95.86 ms** | **5.31×** | 5,341 tok/s | 0.187 ms/tok |

### 3. Execution Phase Breakdown ($N = 512$)

```text
Phase 1: Input Norm & Pre-Projections       :   1.21 ms ( 1.3%)
Phase 2: QKV & Gate Projections (Q4_K MMQ)  :  17.03 ms (17.8%)
Phase 3: SSM Prep, Conv1D & Gating          :   0.89 ms ( 0.9%)
Phase 4: Chunkwise GDN Core (8 × C64)       :  14.28 ms (14.9%)
Phase 5: SSM Norm & SSM Out Proj (Q5_K MMQ) :   7.32 ms ( 7.6%)
Phase 6: Post-Attn Norm & FFN Gate/Up (Q4_K):  38.54 ms (40.2%)
Phase 7: SwiGLU, FFN Down (Q6_K) & Output   :  16.59 ms (17.3%)
Total Layer Execution Time                  :  95.86 ms (100.0%)
```

## Scaling Invariant Analysis

The execution time per token decreases monotonically as prompt length scales:
$$0.322\text{ ms/tok } (N=64) \longrightarrow 0.223\text{ ms/tok } (N=128) \longrightarrow 0.187\text{ ms/tok } (N=512)$$

Unlike V1, there are zero batch boundary cliffs, zero route fallbacks, and zero pathological slowdowns.

## Outcome Gate

**`V2_RECURRENT_ARCHITECTURE_QUALIFIED`**

The single-layer slice achieves a **5.31× speedup** over the canonical oracle at $N=512$ with full numerical validity and clean invariant semantics, surpassing the 1.5× qualification gate.

## Decision

**KEEP** as the foundational building block for Prefill V2.

## Next PRIMARY

Implement one complete repeating 4-layer topology block (`3 × GDN + 1 × GQA`) using the V2 execution contract.
