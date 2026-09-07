# M8 — Latency Attribution and the gfx906 Decode Floor

**Target Model:** `Qwen3.8-27B-Q4_K_M.gguf` (SHA-256: `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`)  
**Target Hardware:** AMD Instinct MI50 32GB (gfx906 / Vega20, 60 CUs, Wave64, 4096-bit HBM2)  
**Qualified Operating Point:** MANUAL DPM Level 7 (1606 MHz SCLK), Level 2 (1000 MHz MCLK), 225.0W Cap  
**Baseline Commit:** `9f20274` (M7 Final Production/Performance Baseline)  
**Date:** September 7, 2026  
**Status:** M8-A Re-qualification Complete | M8-B Attribution Complete | M8-C Floor Defined  

---

## 1. Executive Summary

Milestone M8 establishes the post-M7 latency decomposition and the practical gfx906 decode floor for the 27B-parameter hybrid model `Qwen3.8-27B-Q4_K_M.gguf` on the AMD Instinct MI50, and defines the optimization trajectory required to breach **30.00 tok/s** (≤ 33.33 ms/token) and the stretch goal of **31.00 tok/s** (≤ 32.26 ms/token).

### Re-qualified Baseline (Commit `9f20274`)
- **TG64 Throughput:** **28.62 tok/s** (34.942 ms/token)
- **TG128 Throughput:** **28.53 tok/s** (35.057 ms/token)
- **TG256 Throughput:** **28.23 tok/s** (35.420 ms/token)
- **Context Scaling Penalty:** TG64 → TG128: **+0.33%**; TG64 → TG256: **+1.35%**
- **Correctness:** 16-token generation `replay=PASS`, 64-layer observable contract `PASS` (`logits_cosine=0.999581`, argmax rank 1)
- **Hardware Telemetry:** 2,063 samples, 99.0% at 1606 MHz SCLK, 100% at 1000 MHz MCLK, 55.3 °C mean junction, zero thermal or power throttling.

---

## 2. Phase M8-B: Fresh Post-M7 Latency Attribution

Because M7 introduced major structural changes (single-wave K-quant GEMV dispatches, native Q6_K LM head, inter-layer norm fusion, and tiled attention), historical attribution data from prior milestones is invalid. A fresh full-layer instrumentation sweep was performed at position 63 on the qualified baseline.

### 2.1 Model Weight and Memory Inventory

For a single token generation step, the compulsory weight data loaded across the memory bus is:

| Component | Layers | Weight Type | Bytes per Layer | Total Bytes (All Layers) | Share of Total |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **FFN Gate + Up** | 64 | Q4_K | 100,663,296 | 6,442,450,944 | 37.66% |
| **FFN Down** | 64 | Q4_K | 92,274,688 | 5,905,580,032 | 34.52% |
| **Recurrent QKV Proj** | 48 | Q4_K | 25,165,824 | 1,207,959,552 | 7.06% |
| **Recurrent Gate Proj** | 48 | Q4_K | 16,777,216 | 805,306,368 | 4.71% |
| **Recurrent SSM Out Proj** | 48 | Q4_K | 25,165,824 | 1,207,959,552 | 7.06% |
| **Recurrent Conv / State Weights** | 48 | FP16/FP32 | ~1,500,000 | ~72,000,000 | 0.42% |
| **Attention Q Proj** | 16 | Q4_K | 25,165,824 | 402,653,184 | 2.35% |
| **Attention K Proj** | 16 | Q4_K | 4,194,304 | 67,108,864 | 0.39% |
| **Attention V Proj** | 16 | Q4_K | 4,194,304 | 67,108,864 | 0.39% |
| **Attention Gate + Out Proj**| 16 | Q4_K | 25,165,824 | 402,653,184 | 2.35% |
| **LM Head** | 1 | Q6_K | 525,994,464 | 525,994,464 | 3.07% |
| **Total Compulsory Weights** | - | - | - | **17,106,775,008 B (15.932 GiB)** | **100.00%** |

### 2.2 Execution Stage Latency Breakdown (Per-Token)

Under detailed per-stage timing (`m8b-profile64.log`), the time spent across the functional stages is:

```text
Total Step Time (Unprofiled Streamlined): 34.94 ms
├── FFN Execution (64 Layers):                  24.13 ms (69.0%)
│   ├── FFN Gate + Up SwiGLU:                   12.66 ms (36.2%)
│   └── FFN Down Projection:                    11.47 ms (32.8%)
├── Recurrent / Attention Projections:           8.45 ms (24.2%)
│   ├── Recurrent QKV + Gate + Out (48 layers):  6.91 ms (19.8%)
│   └── Attention Q + K + V + Out (16 layers):   1.54 ms (4.4%)
├── DeltaNet Recurrent Core (48 Layers):         2.45 ms (7.0%)
│   ├── Conv1D + Head Norm:                      0.85 ms (2.4%)
│   └── Recurrent State Update:                  1.60 ms (4.6%)
├── LM Head Projection (Q6_K GEMV):              1.91 ms (5.5%)
├── Tiled Online Softmax Attention (16 Layers):  0.91 ms (2.6%)
├── Activation Quantization (Q8_1 x 337 calls):  1.82 ms (5.2%)
├── Residual Additions & RMSNorms:               0.68 ms (1.9%)
└── Argmax Reduction:                            0.04 ms (0.1%)
```
*(Note: Per-kernel dispatches overlap across streams and memory stages; sum of isolated components reflects raw execution footprint).*

---

## 3. Phase M8-C: The gfx906 Practical Latency Floor

### 3.1 Hardware Memory Bandwidth Upper Bounds

The AMD Instinct MI50 features 32 GB of HBM2 across a 4096-bit interface running at 1000 MHz (2.0 Gbps per pin effective).
- **Theoretical Peak Bandwidth:** `(4096 bits * 2.0 Gbps) / 8 = 1,024 GB/s` (1.024 TB/s).
- **Empirical Streaming Ceiling (BabelStream / Raw Copy):** `~880 GB/s` (86.0% efficiency).
- **Best Measured K-Quant GEMV Bandwidth on Vega20:** `~588 - 605 GB/s` (57.4% - 59.1% of theoretical peak).

### 3.2 Bandwidth-Bound Decode Floors for 15.932 GiB Compulsory Weight Transfer

| Memory Bandwidth | Time for 17.107 GB Weights | Corresponding Throughput Floor | Feasibility on MI50 / gfx906 |
| :---: | :---: | :---: | :--- |
| **1,024 GB/s (100% Theoretical)** | 16.71 ms | 59.8 tok/s | Physical upper bound; unreachable due to DRAM protocol/turnaround overhead. |
| **850 GB/s (Sustained Peak Copy)** | 20.12 ms | 49.7 tok/s | Unreachable for GEMV with dequantization and strided weight blocks. |
| **650 GB/s (Aggressive Kernel Target)**| 26.32 ms | 38.0 tok/s | Achievable only with perfectly repacked contiguous 128-byte vector reads. |
| **600 GB/s (Practical Optimized Ceiling)** | 28.51 ms | 35.1 tok/s | The practical asymptotic decode ceiling on MI50 for batch=1. |
| **588 GB/s (Current MIInfer Peak GEMV)** | **29.09 ms** | **34.4 tok/s** | Current measured sustained bandwidth in FFN Down / LM Head. |

### 3.3 The "Latency Tax" Above the Memory Floor

At the current 34.94 ms/token (28.62 tok/s):
- **Compulsory Weight Load Time (at 588 GB/s):** 29.09 ms (83.3%)
- **Overhead / Non-Weight Bound Latency:** **5.85 ms (16.7%)**

This 5.85 ms overhead consists of:
1. **Activation Quantization (Q8_1):** ~1.82 ms/token.
   - 337 dispatches of `launch_q8_1_quantize_f32` per token.
   - Legacy implementation launched with 32 threads (half-wave) using shared memory and barriers.
2. **DeltaNet Recurrent Core Arithmetic:** ~2.45 ms/token.
   - FP32 state matrix updates, cumulative sum calculations, and chunked recurrent decay logic.
3. **Dispatch Bubbles & Stream Synchronization:** ~0.90 ms/token.
   - CPU-to-GPU command buffer push latency across 1,300+ dispatches (even with HIP graphs, minor intra-graph barrier dependencies exist).
4. **Intermediate Memory Traffic & Normalization:** ~0.68 ms/token.
   - Residual additions, intermediate activation buffers, and un-fused handoffs.

### 3.4 Target Budget for 30.00 tok/s

To reach the Primary Gate:
- **Target Decode Latency:** **≤ 33.33 ms/token**
- **Required Latency Reduction:** `34.942 ms - 33.333 ms = 1.609 ms/token`
- **Stretch Gate Target (31.00 tok/s):** **≤ 32.258 ms/token** (`2.684 ms/token` reduction required).

---

## 4. Candidate Optimization Roadmap for M8

To recover the required 1.61 - 2.68 ms/token, we identify 5 targeted optimization lanes:

### Lane 1: Wave64-Native Q8_1 Quantizer
- **Mechanism:** Replace 32-thread shared-memory quantizer with 256-thread (4 waves) Wave64-native kernel using intra-wave shuffle reductions (`__shfl_xor`) across half-waves. Eliminates all LDS and barriers.
- **Microbenchmark Speedup:** 1.35x (7.32 µs → 5.43 µs on 17408 dims; 6.63 µs → 5.22 µs on 5120 dims).
- **Expected Token-Level Recovery:** **~0.54 ms/token** (337 calls x ~1.6 µs).

### Lane 2: Fused Stage-2 Attention Epilogue Q8_1 Handoff
- **Mechanism:** In the 16 attention layers, fuse the Q8_1 quantization of the gated attention output directly into the `qwen3_splitk_stage2_kernel` epilogue, avoiding an intermediate FP32 VRAM store and separate quantization launch.
- **Expected Token-Level Recovery:** **~0.40 ms/token** (16 layers x ~25 µs).

### Lane 3: DeltaNet Recurrent Core Arithmetic & Vectorization Polish
- **Mechanism:** Vectorize the recurrent state update inner loop with `half2`/`float2` registers and align state decay multiplication to bypass redundant memory transactions.
- **Expected Token-Level Recovery:** **~0.45 ms/token**.

### Lane 4: FFN SwiGLU → Down Direct Quantized Handoff
- **Mechanism:** In the fused gate+up SwiGLU kernel, directly output `Q8_1Block` format into the intermediate buffer for FFN Down projection, bypassing separate FP32 activation write and subsequent Q8_1 quantization.
- **Expected Token-Level Recovery:** **~0.60 ms/token** (64 layers x ~9 µs).

### Lane 5: LM Head Fused Tile Reduction
- **Mechanism:** Streamline the final Q6_K LM Head projection epilogue and top-1 argmax reduction.
- **Expected Token-Level Recovery:** **~0.15 ms/token**.

**Cumulative Potential Latency Recovery:** `0.54 + 0.40 + 0.45 + 0.60 + 0.15 = 2.14 ms/token`  
Projected Decoded Time: `34.94 - 2.14 = 32.80 ms/token` -> **~30.49 tok/s** (Passing the Primary Success Gate).

---

## 5. Conclusion & Action Plan

Phase M8-B and M8-C demonstrate that the 30.00 tok/s target is both physically and practically achievable on MI50/gfx906. Compulsory weight bandwidth at 588 GB/s establishes a decode floor of 29.09 ms (34.4 tok/s). Recovering 1.61 ms from the 5.85 ms non-weight overhead will reach 30.00 tok/s without requiring any speculative model changes.

Execution begins immediately with Candidate Lane 1.
