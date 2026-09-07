# EXP-0191: Fused Residual RMS Norm with Direct Q8_1 Handoff

## Hypothesis

In Qwen3.5-27B decode, standalone `launch_q8_1_quantize_f32` is called after residual RMS norm at:
1. FFN input (Stage 10 in all 64 layers: 64 calls/token)
2. Inter-layer hidden state input (Stage 13/14 in layers 0..62: 63 calls/token)
3. LM head input (Final norm in layer 63: 1 call/token)

Fusing Q8_1 quantization directly into `launch_qwen3_fused_add_rms_norm` will eliminate 128 kernel dispatches and intermediate FP32 VRAM round-trips per token, saving ~0.3 - 0.5 ms/token decode latency.

## Mechanism

In `gfx906/kernels/qwen3_primitives.hip`, `qwen3_fused_add_rms_norm_vec4_kernel` was templated with `<bool EmitQ8>`.
- In-register Q8_1 quantization over 5120 dimensions using intra-wave butterfly shuffle reduction (`__shfl_xor` with offsets 4, 2, 1 across 8-thread sub-wave groups).
- 0 LDS allocations, 0 barriers.
- Coalesced 32-bit integer stores into `q8_out`.
- Activated via runtime environment flag `MIINFER_FUSED_NORM_Q8=1`.

## Baseline (Control)

- Commit: Post-EXP-0190 baseline
- Environment:
  - `MIINFER_Q4K_NATIVE_DOWN=1`
  - `MIINFER_Q4K_NATIVE_GATE_UP=1`
  - `MIINFER_Q4K_NATIVE_Q=1`
  - `MIINFER_Q4K_NATIVE_ATTN_GATE=1`
  - `MIINFER_Q4K_NATIVE_ATTN_OUT=1`
  - `MIINFER_Q4K_NATIVE_K=1`
  - `MIINFER_Q5K_NATIVE_SSM_OUT=1`
  - `MIINFER_KQUANT_NATIVE_QKV=1`
  - `MIINFER_KQUANT_NATIVE_V=1`
  - `MIINFER_Q6K_NATIVE_DOWN=1`
  - `MIINFER_HIP_GRAPH=1`
  - `MIINFER_FUSED_GATE_UP_SWIGLU=1`
  - `MIINFER_FUSED_RECURRENT_CORE=1`
  - `MIINFER_FAST_ARGMAX=1`
  - `MIINFER_TILED_ONLINE_ATTENTION=1`
  - `MIINFER_FUSED_ROPE_NORM=1`
  - `MIINFER_FUSED_ADD_RMS_NORM=1`
  - `MIINFER_FUSED_INTERLAYER_NORM=1`
  - `MIINFER_Q6K_NATIVE_LM_HEAD=1`
  - `MIINFER_WAVE64_Q8=1`
  - `MIINFER_FUSED_CORE_Q8=1`
  - `MIINFER_FUSED_NORM_Q8=0`

## Candidate

- Baseline + `MIINFER_FUSED_NORM_Q8=1` (fusing Q8_1 quantization into residual RMS norm and interlayer norm).

## Environment

- Hardware: AMD Instinct MI50 32GB HBM2 (gfx906 / Vega20, 60 CUs, Wave64)
- DPM Clock State: MANUAL SCLK level 7 (1606 MHz), MCLK level 2 (1000 MHz), 225W power cap
- ROCm version: 7.1.0
- Model: `Qwen3.8-27B-Q4_K_M.gguf`
- Fixture: `/tmp/m6a273-reference-p12`
- Telemetry: Continuous 250ms hardware sampling (`scripts/sample-gpu.sh`)

## Correctness

- 21 / 21 CTests PASS (100%).
- Numerical reference test (`qwen3-primitives-gpu`): exact bitwise match with reference `launch_q8_1_quantize_f32`.
- 64-layer observable contract (`--prefix64-observable-contract`):
  - 64/64 teacher-forced argmax tokens match: PASS.
  - Position 64 `logits_cosine = 0.999581` (>= 0.9995): PASS.
  - Winner rank 1 on GPU and Reference: PASS.
  - Allocations during decode: 0.
  - Replay test: PASS.

## Results

Interleaved A/B benchmark (5 pairs, 5s cooldown, locked 1606/1000 MHz):

### TG64

| Run Pair | Control (tok/s) | Candidate (tok/s) | Delta (tok/s) |
| :--- | :---: | :---: | :---: |
| Pair 1 | 28.8475 | 28.7997 | -0.0478 |
| Pair 2 | 28.8523 | 28.7580 | -0.0943 |
| Pair 3 | 28.8028 | 28.7121 | -0.0907 |
| Pair 4 | 28.8687 | 28.7738 | -0.0949 |
| Pair 5 | 28.8523 | 28.7666 | -0.0857 |
| **Median** | **28.8523 tok/s (34.659 ms)** | **28.7666 tok/s (34.763 ms)** | **-0.0857 tok/s (-0.30%, +0.103 ms)** |

- SCLK stability: 99.9% samples at 1606 MHz (Avg Temp: 49.5°C, Avg Power: 98.9W).

### TG128

| Run Pair | Control (tok/s) | Candidate (tok/s) | Delta (tok/s) |
| :--- | :---: | :---: | :---: |
| Pair 1 | 28.7880 | 28.5015 | -0.2865 |
| Pair 2 | 28.7456 | 28.6289 | -0.1167 |
| Pair 3 | 28.6833 | 28.6271 | -0.0562 |
| Pair 4 | 28.6941 | 28.5911 | -0.1030 |
| Pair 5 | 28.6756 | 28.5766 | -0.0990 |
| **Median** | **28.6941 tok/s (34.850 ms)** | **28.5911 tok/s (34.976 ms)** | **-0.1030 tok/s (-0.36%, +0.126 ms)** |

- SCLK stability: 98.2% samples at 1606 MHz (Avg Temp: 56.6°C, Avg Power: 130.3W).

## Interpretation

1. In HIP Graph replay mode (`MIINFER_HIP_GRAPH=1`), kernel launch latency is ~0 µs because command buffers are pre-baked into the graph engine.
2. `qwen3_fused_add_rms_norm_vec4_kernel` runs on a single workgroup (256 threads) to perform a global sum-of-squares reduction across 5120 elements.
3. Keeping all 20 elements (5 x `float4`) live in VGPRs per thread throughout the residual addition, reduction, and subsequent Q8_1 quantization increased register pressure and limited the instruction scheduling flexibility of the single CU running the workgroup.
4. In contrast, the standalone `launch_q8_1_quantize_f32` (optimized in EXP-0189) dispatches 160 blocks across multiple CUs, achieving high parallelism and cache locality with minimal VGPR pressure.
5. Consequently, the fused kernel increased total execution time by ~100 µs per token.

## Decision

**REJECT** for the primary execution path. Keep `MIINFER_FUSED_NORM_Q8=0` as the default baseline.

## Follow-up

Proceed to **Candidate Lane 4**: Optimize `q4k_wave_fused_gate_up_swiglu_kernel` by replacing the 2048-byte LDS tree reduction and 8 `__syncthreads()` with Wave64 intra-wave shuffle reduction (`__shfl_down`) and single-barrier cross-wave handoff (16 bytes LDS, 1 barrier).
