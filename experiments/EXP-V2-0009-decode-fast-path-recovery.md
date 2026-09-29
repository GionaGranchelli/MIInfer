# EXP-V2-0009: Decode Fast-Path Recovery Inside Prefill V2

## Question
Can we recover the proven single-token decode efficiency of MIInfer on AMD Instinct MI50 (gfx906, Wave64) within the unified `PrefillV2Model` runtime by transplanting historical specialized Wave kernels, shared Q8_1 quantization, and fused SwiGLU paths, while preserving the zero-copy state handoff, HIP Graph replay, and exact numerical trajectory?

## Baseline (V2-0008 Control)
- Hardware: 1 x AMD Instinct MI50 32GB (gfx906:sramecc+:xnack-, 60 CUs, Wave64, 1606 MHz SCLK, 1000 MHz MCLK, 225W)
- Model: `Qwen3.8-27B-Q4_K_M.gguf` (64 layers, 48 GDN SSM + 16 GQA attention)
- V2-0008 Decode Latencies:
  - P64 / TG128: 55.36 ms/step (18.1 tok/s)
  - P512 / TG128: 56.20 ms/step (17.8 tok/s)
  - P2048 / TG128: 57.39 ms/step (17.4 tok/s)
- Static VRAM: 18.17 GiB (KV=32K)

## Fast-Path Differential Analysis

| Family | Historical ~32ms Path | V2-0008 Control | V2-0009 Recovered | Classification |
|:---|:---|:---|:---|:---|
| **Input RMSNorm** | `launch_qwen3_rms_norm` | `launch_qwen3_rms_norm` | `launch_qwen3_rms_norm` | `SAME_FAST_PATH` |
| **Recurrent QKV & Gate** | `launch_q4k_wave_gemv` (shared Q8_1) | Separate `launch_mx_q4k_repacked_mmq` | `launch_q4k_wave_gemv` (shared Q8_1) | `RECOVERED_FAST_PATH` |
| **Beta/Alpha Preparation** | `launch_qwen35_f32_dual_gemm_batch` | `launch_qwen35_f32_dual_gemm_batch` | `launch_qwen35_f32_dual_gemm_batch` | `SAME_FAST_PATH` |
| **Recurrent Conv + L2 Norm**| `launch_qwen35_conv_silu_split_dynamic` | `launch_qwen35_conv_silu_split_dynamic` | `launch_qwen35_conv_silu_split_dynamic` | `SAME_FAST_PATH` |
| **GDN Step Update** | `launch_qwen35_deltanet_state_update` | `launch_qwen35_deltanet_state_update` | `launch_qwen35_deltanet_state_update` | `SAME_FAST_PATH` |
| **SSM Postprocess** | `launch_m12_gdn_postprocess` | `launch_m12_gdn_postprocess` | `launch_m12_gdn_postprocess` | `SAME_FAST_PATH` |
| **SSM Out Projection** | `launch_q5k_wave_gemv` | `launch_mx_q5k_repacked_mmq` | `launch_q5k_wave_gemv` | `RECOVERED_FAST_PATH` |
| **Residual + Post Norm** | `launch_qwen3_fused_add_rms_norm` | `launch_qwen3_fused_add_rms_norm` | `launch_qwen3_fused_add_rms_norm` | `SAME_FAST_PATH` |
| **FFN Gate/Up + SwiGLU** | `launch_q4k_wave_fused_gate_up_swiglu_paired` | 2x `launch_mx_q4k_repacked_mmq` + `silu_mul` | `launch_q4k_wave_fused_gate_up_swiglu_paired` | `RECOVERED_FAST_PATH` |
| **FFN Down Projection** | `mx_repacked_mmv_kernel` (Q6_K) | `mx_repacked_mmv_kernel` (Q6_K) | `mx_repacked_mmv_kernel` (Q6_K) | `SAME_FAST_PATH` (Proven Best for 13824) |
| **Attention Q/K/V** | `launch_q4k_wave_gemv` (shared Q8_1) | Separate `launch_mx_q4k_repacked_mmq` | `launch_q4k_wave_gemv` (shared Q8_1) | `RECOVERED_FAST_PATH` |
| **Attention RoPE & KV Store**| `launch_qwen35_fused_..._dynamic` | `launch_qwen35_fused_..._dynamic` | `launch_qwen35_fused_..._dynamic` | `SAME_FAST_PATH` |
| **Split-K GQA Attention** | `launch_qwen35_tiled_online_attention_f16_dynamic` | `launch_qwen35_tiled_online_attention_f16_dynamic` | `launch_qwen35_tiled_online_attention_f16_dynamic` | `SAME_FAST_PATH` |
| **Attention O Projection** | `launch_q4k_wave_gemv` | `launch_mx_q4k_repacked_mmq` | `launch_q4k_wave_gemv` | `RECOVERED_FAST_PATH` |
| **LM Head** | `launch_q6k_wave_gemv` | `launch_qwen3_q6_k_q8_k_gemv` | `launch_q6k_wave_gemv` | `RECOVERED_FAST_PATH` |
| **Device Argmax** | `launch_qwen3_argmax` | `launch_qwen3_argmax` | `launch_qwen3_argmax` | `SAME_FAST_PATH` |
| **Execution Capture** | Reusable `hipGraphExec_t` | Reusable `hipGraphExec_t` | Reusable `hipGraphExec_t` | `SAME_FAST_PATH` |

## Micro-Bakeoff Results

```text
Bakeoff 1: FFN Gate + Up + SwiGLU (13824 x 5120)
- Control V2 (Mx MMQ 2x + SiLU)   : 209.82 us
- Candidate A (Fused Paired Wave)  : 144.40 us -> SPEEDUP: 1.45x (Saves 65.43 us/layer, 4.19 ms/token)

Bakeoff 2: SSM-Out Projection (Q5_K 5120 x 5120)
- Control V2 (Mx MMV)              :  72.52 us
- Candidate C (Wave GEMV)          :  59.72 us -> SPEEDUP: 1.21x (Saves 12.81 us/layer, 0.61 ms/token)

Bakeoff 3: FFN Down Projection (Q6_K 5120 x 13824)
- Control V2 (Mx MMV)              : 122.94 us
- Candidate E1 (Q8_1 MMVQ)         : 166.73 us -> SPEEDUP: 0.74x
- Candidate E2 (Q8_K GEMV)         : 381.35 us -> SPEEDUP: 0.32x
Decision on FFN Down: Retain Mx MMV as optimal for non-1024-aligned Q6_K width.
```

## Cumulative Performance Progress

| Step | P64 Step ms (tok/s) | P512 Step ms (tok/s) | P2048 Step ms (tok/s) | Delta vs Control | Static VRAM | Free Headroom |
|:---|---:|---:|---:|---:|---:|---:|
| **V2-0008 Control** | 55.36 ms (18.1 tok/s) | 56.20 ms (17.8 tok/s) | 57.39 ms (17.4 tok/s) | baseline | 20.37 GiB | 11.63 GiB |
| **+ Candidate A, C, D** (Wave FFN SwiGLU, Projections) | 48.61 ms (20.6 tok/s) | 49.43 ms (20.2 tok/s) | 50.61 ms (19.8 tok/s) | -6.77 ms/tok (+13.5%) | 29.67 GiB | 2.33 GiB |
| **+ Candidate E** (Wave Q6_K LM Head) | **44.24 ms (22.6 tok/s)** | **44.73 ms (22.4 tok/s)** | **45.95 ms (21.8 tok/s)** | **-11.47 ms/tok (+25.8%)** | 29.74 GiB | 2.26 GiB |

## Context Scaling Verification

- $P64 \text{ Decode}: 44.24\text{ ms/step}$
- $P512 \text{ Decode}: 44.73\text{ ms/step}$ ($+1.1\%$)
- $P2048 \text{ Decode}: 45.95\text{ ms/step}$ ($+3.9\%$)
- Ratio $P2048 / P64 = 1.0387\times \le 1.10\times$ (Flat context scaling preserved).

## Correctness Qualification
1. **Zero-Copy State Hand-off**: Step 1..16 token IDs match control bit-identically (`[7676, 220, 15, 25, 220, 15, 220, 19, 271, 220, 15, 11, 220, 198, 7676, 11]`).
2. **Multi-Turn Determinism**: Turn 1 vs Turn 3 bit-identical matching across 16 generated tokens following state reset.
3. **Prefill Preservation**:
   - P64 TTFT: 617.10 ms (vs 613.19 ms)
   - P512 TTFT: 2310.71 ms (vs 2304.45 ms)
   - P2048 TTFT: 9797.28 ms (vs 9709.18 ms)
4. **Unit Tests**: 24/24 CTest regression tests passing (100%).

## Decision
**KEEP**. Passed Minimum Gate (<45 ms/token). Reached 44.24 - 44.73 ms/token (22.4 - 22.6 tok/s), achieving a +25.8% throughput increase over V2-0008.
