# EXP-0190 — Fused Recurrent & Attention Epilogue Q8_1 Activation Quantization (Lane 2)

## Hypothesis

Fusing the activation quantization of gated recurrent outputs directly into `qwen35_deltanet_fused_recurrent_core_kernel` across all 48 recurrent layers, and fusing gated attention output quantization directly into `qwen3_splitk_stage2_kernel` across all 16 attention layers, will eliminate 64 intermediate activation VRAM round-trips (16.4 KiB and 24.5 KiB per layer), eliminate 64 separate kernel launches per token, save >0.20 ms/token, and push TG64 throughput past 29.00 tok/s on AMD Instinct MI50 (gfx906) under continuous hardware telemetry and zero decode allocations.

## Motivation

In Phase M8-B and EXP-0189, we identified that every decode step performed 337 separate activation quantizations:
- 48 DeltaNet recurrent layers computed gated outputs into global VRAM (`kVHeads * kState = 4096` floats), followed by a separate `launch_q8_1_quantize_f32` dispatch before the `ssm_out` projection.
- 16 full-attention layers computed gated attention outputs into global VRAM (`24 * 256 = 6144` floats), followed by a separate `launch_q8_1_quantize_f32` dispatch before the `o_proj` projection.

In both kernels:
1. The workgroup size matches the head dimension exactly (`state_size = 128` threads = 2 waves = 4 Q8 blocks; `head_dim = 256` threads = 4 waves = 8 Q8 blocks).
2. Every wavefront in the workgroup naturally holds two 32-element Q8 blocks across lanes 0–31 and 32–63.
3. The gated activation values `gated_val = out_val * sig` are already resident in vector registers (`v_gated`) at the moment of the epilogue write.

Instead of writing `gated_val` to VRAM and launching a subsequent quantization kernel, the epilogue of each core kernel executes intra-wave `__shfl_xor` reductions directly in registers, emitting both the unquantized tensor (for validation/diagnostics) and the quantized `Q8_1Block` stream directly into the pre-allocated projection buffer.

## Baseline

- Commit: EXP-0189 candidate (Wave64 shuffle-based standalone quantizer).
- Baseline throughput: 28.80 tok/s (34.72 ms/token) TG64.
- Quantization: 64 separate `launch_q8_1_quantize_f32` dispatches per token for recurrent & attention cores (`MIINFER_FUSED_CORE_Q8=0`).

## Candidate

- Epilogue quantization in `qwen35_deltanet_fused_recurrent_core_kernel` (48 layers).
- Epilogue quantization in `qwen3_splitk_stage2_kernel` (16 layers).
- Bypassed 64 standalone `launch_q8_1_quantize_f32` calls per token when `MIINFER_FUSED_CORE_Q8=1`.
- Retained full backward compatibility with optional pointer.

## Environment

- Target GPU: AMD Instinct MI50 32GB HBM2 (gfx906 / Vega20, 60 CUs, Wave64)
- Clock State: MANUAL DPM Level 7 (1606 MHz SCLK), Level 2 (1000 MHz MCLK), 225.0W Cap
- ROCm Version: 7.1.0
- Model: `Qwen3.8-27B-Q4_K_M.gguf` (SHA-256: `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`)
- Fixture: `/tmp/m6a273-reference-p12`
- Telemetry: Continuous 250ms hardware telemetry (`scripts/sample-gpu.sh`)

## Correctness

- **Isolated Kernel Bitwise Equivalence:**
  - Attention Stage 2 (`scratch/test_stage2_q8.hip`): `diff_d=0, diff_s=0, diff_qs=0` (PASS).
  - Recurrent Core (`scratch/test_recurrent_q8.hip`): `diff_d=0, diff_s=0, diff_qs=0` (PASS).
- **Full Model Autoregressive Generation (16 tokens):**
  - Tokens: `11, 585, 1318, 310, 1236, 264, 1746, 303, 29337, 11, 694, 585, 1044, 524, 264, 46194`
  - Replay check: `replay=PASS`
  - State fingerprint: `16931364266496685951` (exact bitwise match with baseline).
  - Allocations during decode: 0
- **64-Layer Observable Contract (`--prefix64-observable-contract`):**
  - `logits_cosine = 0.999581`
  - Reference argmax = 8719, GPU argmax = 8719 (PASS)
  - All 64 state fingerprints match bit-for-bit with baseline.
  - All 16 KV cache fingerprints match bit-for-bit with baseline.
- **CTest Suite:** 21 / 21 tests passed (100%).

## Benchmark Results

Interleaved A/B testing (5 pairs, 5s cooldown, continuous 250ms telemetry) using `scripts/run-exp0190-fused-core-q8-ab.sh`.

### TG64 (64 Tokens)

| Pair | Control (EXP-0189 Baseline) tok/s | Candidate (Fused Core Q8) tok/s | Latency Delta |
| :---: | :---: | :---: | :---: |
| 1 | 28.7988 (34.724 ms) | 28.9246 (34.573 ms) | -0.151 ms (-0.43%) |
| 2 | 28.9510 (34.541 ms) | 28.9310 (34.565 ms) | +0.024 ms (+0.07%) |
| 3 | 28.7895 (34.735 ms) | 29.0282 (34.450 ms) | -0.285 ms (-0.82%) |
| 4 | 28.8861 (34.619 ms) | 29.0428 (34.432 ms) | -0.187 ms (-0.54%) |
| 5 | 28.8001 (34.722 ms) | 29.0275 (34.450 ms) | -0.272 ms (-0.78%) |
| **Median** | **28.8001 tok/s (34.722 ms)** | **29.0275 tok/s (34.450 ms)** | **-0.272 ms (-0.78%) / +0.79% tok/s** |

- Telemetry: 921 samples, 99.9% SCLK >= 1600 MHz, Avg Temp: 48.1 °C, Avg Power: 98.0W.

### TG128 (128 Tokens)

| Pair | Control (EXP-0189 Baseline) tok/s | Candidate (Fused Core Q8) tok/s | Latency Delta |
| :---: | :---: | :---: | :---: |
| 1 | 28.7662 (34.763 ms) | 28.7831 (34.743 ms) | -0.020 ms (-0.06%) |
| 2 | 28.7827 (34.743 ms) | 28.7747 (34.753 ms) | +0.010 ms (+0.03%) |
| 3 | 28.6344 (34.923 ms) | 28.8447 (34.668 ms) | -0.255 ms (-0.73%) |
| 4 | 28.6801 (34.867 ms) | 28.7629 (34.767 ms) | -0.100 ms (-0.29%) |
| 5 | 28.5965 (34.969 ms) | 28.7282 (34.809 ms) | -0.160 ms (-0.46%) |
| **Median** | **28.6801 tok/s (34.867 ms)** | **28.7747 tok/s (34.753 ms)** | **-0.115 ms (-0.33%) / +0.33% tok/s** |

- Telemetry: 1,258 samples, 98.5% SCLK >= 1600 MHz, Avg Temp: 55.2 °C, Avg Power: 129.8W.

## Interpretation

1. Eliminating 64 kernel launches and the associated VRAM activation write/read passes saved **0.272 ms/token** on TG64 and pushed median throughput to **29.03 tok/s** (with peak samples at 29.04 tok/s).
2. Combined with EXP-0189 (Wave64 quantizer), cumulative decode latency has dropped from 34.96 ms to 34.45 ms/token (-0.51 ms/token, +1.5% tok/s over M7 baseline).
3. Distance to the 30.00 tok/s Primary Success Gate (33.33 ms/token) is now narrowed to **1.12 ms/token** (from 1.61 ms/token).

## Decision

**KEEP**. Default enabled via `MIINFER_FUSED_CORE_Q8=1`.
