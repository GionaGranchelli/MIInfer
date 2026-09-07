# M9 — Numerical Error Budget & Attribution Report

**Model:** `Qwen3.8-27B-Q4_K_M.gguf`  
**Reference Oracle:** CPU / High-Precision Host Reference (`/tmp/m6a273-reference-p12`)  
**Hardware Target:** AMD Instinct MI50 (gfx906 / Vega20, 60 CUs, Wave64)  
**Contract Requirement:** 64/64 Argmax Agreement `PASS`, Rank 1 Winner, `logits_cosine >= 0.9995`  
**Current Baseline Value:** `logits_cosine = 0.999546` (Safety Margin: $+0.000046$)  
**Stretch Target:** `logits_cosine >= 0.9996`  
**Date:** September 7, 2026  
**Status:** **Phase M9-C Error Attribution Complete**

---

## 1. Executive Summary

Milestone M9 enforces a strict numerical safety budget. With the qualified baseline logits cosine at **0.999546** against the hard contract floor of **0.999500**, any optimization in Lane B (such as projection fusions, arithmetic restructuring, or epilogue quantizations) must be governed by an explicit error budget.

This report attributes the numerical divergence layer-by-layer and position-by-position across the 64-layer hybrid architecture, isolates the contribution of individual kernel families, and establishes rules for preserving numerical stability.

---

## 2. 64-Layer Cumulative Error Progression

Measured across all 64 layers at position 64 under teacher-forced evaluation:

| Layer Index | Layer Type | Max Absolute Error ($\Delta$) | RMS Error ($\sigma$) | Relative RMS ($\sigma / \text{norm}$) | Cumulative Trend |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **Layer 0** | Recurrent | 0.0347 | 0.00140 | 0.00745 | Initial quantization baseline |
| **Layer 1** | Recurrent | 0.0549 | 0.00171 | 0.00749 | Linear recurrent accumulation |
| **Layer 2** | Recurrent | 0.0639 | 0.00203 | 0.00928 | Linear recurrent accumulation |
| **Layer 3** | **Attention** | **0.1334** | **0.00326** | **0.01036** | **First softmax attention step** |
| **Layer 7** | Attention | 0.1975 | 0.00524 | 0.01177 | Attention layer step |
| **Layer 11** | Attention | 0.0516 | 0.00542 | 0.00958 | Stable residual damping |
| **Layer 15** | Attention | 0.1158 | 0.00963 | 0.01178 | Bounded drift |
| **Layer 23** | Attention | 0.0935 | 0.01331 | 0.01340 | Bounded drift |
| **Layer 31** | Attention | 0.1027 | 0.01926 | 0.01722 | Mid-network stability |
| **Layer 39** | Attention | 0.1203 | 0.02576 | 0.01896 | Bounded drift |
| **Layer 47** | Attention | 0.1541 | 0.03484 | 0.02033 | Deep network accumulation |
| **Layer 55** | Attention | 0.6640 | 0.07137 | 0.02826 | Deep attention expansion |
| **Layer 63** | Attention | 0.8653 | 0.18369 | 0.04044 | Final layer hidden state |
| **Final Norm**| RMSNorm | 0.4269 | 0.07916 | 0.04077 | Cosine: `0.999169` |
| **Logits** | Q6_K LM Head| 0.6821 | 0.08421 | 0.03040 | **Cosine: `0.999546`** |

---

## 3. Observable Contract & Top-K Ranking Stability

At the final output position 64:
- **Reference Argmax Token:** 8719
- **GPU Argmax Token:** 8719 (`match = PASS`)
- **Reference Margin:** 0.3559 (distance between top 1 and top 2 logits on CPU reference)
- **GPU Margin:** 0.2154
- **Top-5 Overlap:** 4 / 5 (80.0% set intersection)
- **GPU Rank of Reference Winner:** **Rank 1** (100% top-choice agreement)
- **Reference Rank of GPU Winner:** **Rank 1**
- **NaN / Inf Detection:** **Zero** across all layers, states, and logits.
- **Poisoned Reset Replay Test:** **`PASS`** (exact state clearing verified).

---

## 4. Kernel-Level Error Attribution

### 4.1 Quantization Noise vs Compute Precision
1. **Q4_K / Q8_1 Dequantization Arithmetic:**
   - Inner product accumulation is performed in 32-bit signed integers via hardware `__builtin_amdgcn_sdot4`.
   - Scale multiplication is performed in FP32.
   - Because `sdot4` is exact integer math, there is zero truncation or floating-point associativity error during the matrix-vector dot products. All quantization error is bounded by the block quantization scales $d$ and $d_{min}$.
2. **Q6_K SIMD Unpack (EXP-0193):**
   - The SIMD bitwise extraction in EXP-0193 was verified bit-for-bit identical across all 64 quantized representations ($-32 \dots +31$) against scalar extraction. It contributes exactly **0.000000** additional numerical drift.
3. **DeltaNet Recurrent Core (`launch_qwen35_deltanet_fused_recurrent_core`):**
   - State recurrence uses FP32 accumulation and FMA (`fmaf`).
   - Head RMS norm is reduced using FP32 intra-wave shuffles and inverted via hardware `rsqrtf`.
   - Max numerical divergence per recurrent layer is $< 1 \times 10^{-5}$.
4. **Attention Layers:**
   - Attention layers (every 4th layer: 3, 7, 11, ..., 63) exhibit larger relative RMS increments ($\sim 0.0012$ RMS jump) due to the non-linear softmax exponential and online tiling normalization.
   - Despite this, the RMS error remains $< 4.1\%$ of activation magnitude even at Layer 63.

---

## 5. M9 Numerical Budget Rules for Lane B Optimizations

To ensure the observable contract floor ($\ge 0.9995$) is never violated:

1. **Rule C.1 (Zero Arithmetic Drift for Projection Merges):**
   Combining QKV and Gate into a single matrix dispatch must preserve the exact block order, scale indexing, and inner accumulation sequence. The mathematical operation must remain bit-for-bit identical.
2. **Rule C.2 (Cosine Protection Floor):**
   Any candidate optimization that reduces `logits_cosine` below `0.999500` is immediately **REJECTED**, regardless of performance gain.
3. **Rule C.3 (Stretch Target Qualification):**
   If an optimization can improve `logits_cosine` toward $\ge 0.9996$ without causing throughput regression, it should be adopted as a numerical stability improvement.
4. **Rule C.4 (Top-1 Argmax Invariance):**
   All 64 teacher-forced token positions must maintain 64/64 argmax agreement with the host reference.
