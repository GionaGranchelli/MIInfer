# M9 — Post-M8 Latency Floor & Decomposition Report

**Model:** `Qwen3.8-27B-Q4_K_M.gguf` (SHA-256: `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`)  
**Hardware:** AMD Instinct MI50 32GB HBM2 (gfx906 / Vega20, 60 CUs, 4096-bit HBM2 bus)  
**Qualified Clocks:** MANUAL DPM Level 7 (1606 MHz SCLK), DPM Level 2 (1000 MHz MCLK), 225.0W Cap  
**Baseline Commit:** `ff7aeff` (Qualified M8 Primary Gate: 30.29 tok/s / 33.01 ms/token TG64)  
**Target:** **$\ge$ 32.00 tok/s / $\le$ 31.25 ms/token** (Recovery requirement: **1.76 ms/token**)  
**Date:** September 7, 2026  
**Status:** **Phase M9-A Attribution Complete**

---

## 1. Executive Summary

Milestone M9 begins from the qualified Milestone M8 baseline (`30.29 tok/s / 33.01 ms/token` TG64, `29.93 tok/s` TG128). This document establishes the fresh post-M8 latency attribution across all 64 layers of `Qwen3.8-27B-Q4_K_M.gguf`, accounts for all optimizations kept in M8 (EXP-0189 Wave64 Q8 quantizer, EXP-0190 fused recurrent core Q8 epilogue, EXP-0192 SwiGLU intra-wave shuffle reduction, and EXP-0193 vectorized SIMD Q6_K decoding), measures the compulsory memory traffic across HBM2, and constructs a ranked optimization roadmap to recover the 1.76 ms/token needed to achieve $\ge 32.00$ tok/s.

---

## 2. Model Weight & Compulsory Memory Inventory

For every single token generation step, the compulsory weight data streamed across HBM2 is:

| Component | Layers | Quant Type | Dimension ($M \times K$) | Bytes per Layer | Total Bytes (All Layers) | Share |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **FFN Gate + Up** | 64 | Q4_K | $2 \times 8704 \times 5120$ | 100,663,296 | 6,442,450,944 | 37.56% |
| **FFN Down (Layers 1–63)** | 63 | Q4_K | $5120 \times 17408$ | 92,274,688 | 5,813,305,344 | 33.89% |
| **FFN Down (Layer 0)** | 1 | Q6_K | $5120 \times 17408$ | 137,756,672 | 137,756,672 | 0.80% |
| **Recurrent QKV Proj** | 48 | Q4_K | $4096 \times 5120$ | 25,165,824 | 1,207,959,552 | 7.04% |
| **Recurrent SSM Out Proj** | 48 | Q5_K | $2048 \times 10240$ | 25,165,824 | 1,207,959,552 | 7.04% |
| **Recurrent Gate Proj** | 48 | Q4_K | $2048 \times 5120$ | 16,777,216 | 805,306,368 | 4.70% |
| **Attention Q Proj** | 16 | Q4_K | $4096 \times 5120$ | 25,165,824 | 402,653,184 | 2.35% |
| **Attention Out Proj** | 16 | Q4_K | $5120 \times 4096$ | 25,165,824 | 402,653,184 | 2.35% |
| **Attention K Proj** | 16 | Q4_K | $1024 \times 5120$ | 4,194,304 | 67,108,864 | 0.39% |
| **Attention V Proj** | 16 | Q4_K | $1024 \times 5120$ | 4,194,304 | 67,108,864 | 0.39% |
| **LM Head** | 1 | Q6_K | $151936 \times 5120$ | 525,994,464 | 525,994,464 | 3.07% |
| **Conv1D / Norm / Bias** | 64 | FP16/FP32 | Various | ~1,125,000 | ~72,000,000 | 0.42% |
| **Total Compulsory Weights** | - | - | - | - | **17,150,346,992 B (15.972 GiB)** | **100.00%** |

---

## 3. Bandwidth Analysis & Physical Floors

- **Theoretical Peak Bandwidth:** 4096 bits $\times$ 2.0 Gbps / 8 = **1,024.0 GB/s**.
- **Empirical Streaming Copy Peak (BabelStream):** **~880 GB/s** (85.9% efficiency).
- **M8 Baseline Operating Bandwidth:**
  $$\text{Effective Bandwidth} = \frac{17.150\text{ GB}}{0.03301\text{ s}} = \mathbf{519.5\text{ GB/s}} \quad (50.7\%\text{ of theoretical, } 59.0\%\text{ of stream peak})$$
- **M9 Target Operating Bandwidth (at 32.00 tok/s / 31.25 ms):**
  $$\text{Required Bandwidth} = \frac{17.150\text{ GB}}{0.03125\text{ s}} = \mathbf{548.8\text{ GB/s}} \quad (53.6\%\text{ of theoretical, } 62.4\%\text{ of stream peak})$$

Because individual optimized GEMV kernels (such as FFN Down) already sustain **593.0 GB/s** in isolation, an aggregate decode bandwidth of 548.8 GB/s is physically feasible and well within Vega20 HBM2 capability.

---

## 4. Post-M8 Execution Stage Latency Attribution

From the fresh instrumentation sweep on the qualified M8 baseline (`m9a-profile64.log`):

```text
Streamlined HIP Graph Decode Step: 33.01 ms (100.0%)
├── FFN Execution (64 Layers):                  22.95 ms (69.5%)
│   ├── FFN Gate + Up SwiGLU (64 layers):       12.30 ms (37.3%)
│   └── FFN Down Projection (64 layers):        10.65 ms (32.3%)
├── Recurrent Projections & Core (48 Layers):    7.65 ms (23.2%)
│   ├── QKV Projection (48 layers):              2.80 ms (8.5%)
│   ├── SSM Output Projection (48 layers):       2.15 ms (6.5%)
│   ├── Gate Projection (48 layers):             1.25 ms (3.8%)
│   └── Fused Recurrent Core + Conv (48 layers): 1.45 ms (4.4%)
├── Attention Layers (16 Layers):                1.42 ms (4.3%)
│   ├── Q, K, V Projections (16 layers):         0.82 ms (2.5%)
│   ├── Tiled Attention + Epilogue Q8:           0.36 ms (1.1%)
│   └── Attention Out Projection (16 layers):    0.24 ms (0.7%)
├── LM Head Projection (Q6_K SIMD GEMV):         0.85 ms (2.6%)
└── Normalization, Residuals, Argmax, Sync:      0.14 ms (0.4%)
```

### 4.1 Structural Inefficiencies in the Current Execution Path

1. **Fragmented Input Projections on the Same Activation:**
   - In each of the 48 recurrent layers, `d_qkv_native` ($4096 \times 5120$) and `d_attn_gate_native` ($2048 \times 5120$) consume the identical input activation `d_norm` but are dispatched as two independent GEMVs. The 2,048-row Gate launch only provides 1,024 workgroups (17 per CU), suffering from launch tail latency and wave starvation.
   - In each of the 16 attention layers, Q ($4096 \times 5120$), K ($1024 \times 5120$), and V ($1024 \times 5120$) are dispatched as 3 separate kernels. K and V (1,024 rows = 512 workgroups = 8.5 per CU) have severely depressed occupancy.

2. **Redundant FFN Down Activation Quantization:**
   - 64 times per token, `launch_q4k_wave_fused_gate_up_swiglu` writes FP32 activations to VRAM. Standalone `launch_q8_1_quantize_f32` is then dispatched to quantize the 5,120 floats into 160 Q8 blocks before launching FFN Down.
   - Because the SwiGLU workgroup has 256 threads (4 waves of 64 threads), each wave produces 64 rows. Fusing intra-wave `__shfl_xor` Q8 quantization into the SwiGLU epilogue can emit `Q8_1Block` directly, saving 64 kernel launches and round-trip memory traffic.

3. **Tile Arithmetic Redundancy in `Q4KDecoder`:**
   - In `Q4KDecoder::process_part`, `__half2float(q.d)` is converted twice per part and multiplied twice. Factoring `q.d` out of the inner loop reduces instruction pressure in every non-SwiGLU Q4_K projection (FFN Down, QKV, Gate, Q, Out).

---

## 5. Ranked M9 Optimization Opportunities

| Rank | Optimization Candidate | Expected Latency Saved | Target Stage | Mechanism | Implementation Risk | Numerical Risk |
| :---: | :--- | :---: | :--- | :--- | :---: | :---: |
| **1** | **Fused / Combined Recurrent QKV+Gate Projection** | **0.40 – 0.65 ms** | Recurrent Projections | Combine $4096 + 2048 = 6144$ rows into single $6144 \times 5120$ launch. Eliminates 48 kernel launches, raises wave occupancy by 3x on Gate. | Low | Zero |
| **2** | **Fused / Combined Attention Q+K+V Projection** | **0.25 – 0.45 ms** | Attention Projections | Combine $4096 + 1024 + 1024 = 6144$ rows into single launch. Eliminates 32 kernel launches, solves K/V under-occupancy. | Low | Zero |
| **3** | **SwiGLU Epilogue Direct Q8_1 Emission** | **0.30 – 0.50 ms** | FFN Activation | Inline Wave64 `__shfl_xor` Q8_1 quantization directly into `q4k_wave_fused_gate_up_swiglu_kernel` epilogue. Eliminates 64 launches of `q8_1_quantize_f32`. | Low-Med | Zero |
| **4** | **Vectorized Q4_K Inner Arithmetic Factoring** | **0.20 – 0.40 ms** | All Q4_K GEMVs | Factor `__half2float(q.d)` out of `Q4KDecoder::process_part`, parallelize scale/minimum dot products. | Very Low | Zero |
| **5** | **Fused Conv1D + Head RMS Norm Epilogue** | **0.15 – 0.25 ms** | Recurrent Core | Fold Conv1D state shift into the recurrent core prologue or fuse head norm. | Medium | Low |

**Total Recoverable Latency Range:** **1.30 – 2.25 ms/token**.  
Target to reach $\ge 32.00$ tok/s ($\le 31.25$ ms): **1.76 ms/token**.
