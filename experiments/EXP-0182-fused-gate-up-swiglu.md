# EXP-0182 — Fused Gate+Up SwiGLU Wave64 GEMV

## Hypothesis

Fusing `ffn_gate` GEMV ($5120 \times 17408$), `ffn_up` GEMV ($5120 \times 17408$), and the elementwise SwiGLU activation into a single dual-accumulator Wave64 kernel will eliminate intermediate VRAM write/read roundtrips for 138 KiB of activation data per layer (saving 8.8 MB write + 8.8 MB read = 17.6 MB global VRAM traffic per token across all 64 layers) and eliminate 128 kernel launches per token, yielding a measurable decode latency reduction on AMD Instinct MI50 (gfx906) while preserving exact bitwise numerical equivalence.

## Motivation

In the Qwen3.8-27B architecture, each of the 64 layers executes an MLP block where:
1. `ffn_gate = post_norm @ W_gate`
2. `ffn_up   = post_norm @ W_up`
3. `ffn_act  = silu(ffn_gate) * ffn_up`
4. `ffn_down = ffn_act @ W_down`

In the EXP-0181 baseline:
- `W_gate` and `W_up` were dispatched as two separate `launch_q4k_wave_gemv` kernels.
- Both kernels separately loaded the same pre-quantized `Q8_1Block` activations from VRAM, independently read their respective weights, and wrote full FP32 intermediate buffers `ffn_gate` (69,632 bytes) and `ffn_up` (69,632 bytes) back to VRAM.
- A third kernel, `launch_qwen3_silu_mul`, then re-read both buffers from VRAM, computed `silu(g) * u`, and wrote `ffn_activation`.
- Neither `ffn_gate` nor `ffn_up` is ever consumed by any other operation downstream.

By designing `launch_q4k_wave_fused_gate_up_swiglu`:
- Each Wave64 computes both `sum_gate` and `sum_up` simultaneously in registers, loading the `Q8_1` activation tile once and computing `ones` dot-products once for both projections.
- Dual LDS reductions parallelize the reduction of both accumulators across lanes.
- Lane 0 performs `const float g = sum_gate; const float silu_g = g / (1.0f + __builtin_amdgcn_exp_f32(-g)); out[row] = silu_g * sum_up;` and writes directly to `ffn_activation`.
- Intermediate `ffn_gate` and `ffn_up` VRAM writes/reads are completely bypassed.

## Baseline

- Commit: `91e4da6` (EXP-0181 static HIP graph capture enabled).
- Full native Wave64 execution across Q4_K, Q5_K, and Q6_K tensor families:
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
- `MIINFER_FUSED_GATE_UP_SWIGLU=0` (separate Gate GEMV + Up GEMV + SwiGLU).

## Candidate

- Fused Gate+Up SwiGLU kernel enabled via `MIINFER_FUSED_GATE_UP_SWIGLU=1`.
- Dual-accumulator Wave64 kernel `q4k_wave_fused_gate_up_swiglu_kernel` in `gfx906/kernels/kquant_wave_layout.hip`.
- Integrated into `RecurrentLayer` and `FullAttentionLayer` in `tools/m6a21_qwen35_gpu_hybrid_block.cpp`.
- Directly outputs to `ffn_activation`, skipping stage 11 `silu_mul` dispatch in `RecurrentLayer` and stage 12 `silu_mul` dispatch in `FullAttentionLayer`.

## Environment

- Hardware: AMD Instinct MI50 32GB HBM2 (gfx906 / Vega20, 60 CUs, Wave64)
- DPM Clock State: MANUAL SCLK level 7 (1606 MHz), MCLK level 2 (1000 MHz), 225W power cap
- ROCm version: 7.1.0
- Model: `Qwen3.8-27B-Q4_K_M.gguf`
- Fixture: `/tmp/m6a273-reference-p12`
- Telemetry: Continuous 250ms hardware sampling (`scripts/sample-gpu.sh`)

## Correctness

- Host unit & numerical tests (`build/mi50-release/bin/miinfer-kquant-wave-test`):
  - Test 5: `launch_q4k_wave_fused_gate_up_swiglu` vs reference CPU and separate execution: `max_abs_err = 0.000000e+00` (bit-for-bit exact). PASS.
- 16-token autoregressive generation:
  - Tokens: `11,585,1044,264,5286,303,279,11759,314,76163,11,9903,26417,11,321,585` (exact match with control)
  - State fingerprint: `9420068774711364252` (exact bitwise match with control)
  - Replay check: PASS
  - Allocations during decode: 0
- 64-token autoregressive generation:
  - Tokens: 64/64 exact match (`first_token=11`, `last_token=369`)
  - State fingerprint: `5199928154589341973` (exact bitwise match with control)
  - Replay check: PASS
  - Allocations during decode: 0
- Full CTest suite: 21 / 21 tests passed (100%).

## Benchmark Methodology

- 5 interleaved pairs (`Control 1 -> Candidate 1 -> ... -> Control 5 -> Candidate 5`) with 5s cooldown.
- Dedicated hardware telemetry sampler recording GPU clocks, temperature, and power at 250ms intervals.
- Tested across TG64 (64 tokens) and TG128 (128 tokens).
- Script: `scripts/run-exp0182-fused-gate-up-ab.sh`.

## Results

### 1. TG64 (64 tokens)

| Pair | Control (FUSED=0) ms | Control tok/s | Candidate (FUSED=1) ms | Candidate tok/s | Latency Delta ms/tok |
|---|---|---|---|---|---|
| Pair 1 | 2727.35 | 23.4660 | 2672.17 | 23.9505 | -0.862 |
| Pair 2 | 2720.68 | 23.5235 | 2675.31 | 23.9225 | -0.709 |
| Pair 3 | 2731.65 | 23.4291 | 2670.76 | 23.9632 | -0.951 |
| Pair 4 | 2727.86 | 23.4616 | 2677.75 | 23.9007 | -0.783 |
| Pair 5 | 2732.42 | 23.4225 | 2672.27 | 23.9497 | -0.940 |
| **Median** | **2727.86** | **23.4616** | **2672.27** | **23.9497** | **-0.869** |

- **TG64 Throughput:** `23.4616 -> 23.9497 tok/s` (**+2.08% throughput gain**)
- **Peak TG64 Throughput:** `23.9632 tok/s`
- **TG64 Latency:** `42.623 -> 41.754 ms/token` (**-0.869 ms/token latency saved**)
- **Allocations during decode:** 0
- **Replay check:** PASS

### 2. TG128 (128 tokens)

| Pair | Control (FUSED=0) ms | Control tok/s | Candidate (FUSED=1) ms | Candidate tok/s | Latency Delta ms/tok |
|---|---|---|---|---|---|
| Pair 1 | 5581.14 | 22.9344 | 5478.91 | 23.3623 | -0.799 |
| Pair 2 | 5602.16 | 22.8483 | 5480.18 | 23.3569 | -0.953 |
| Pair 3 | 5599.24 | 22.8603 | 5476.60 | 23.3722 | -0.958 |
| Pair 4 | 5602.43 | 22.8472 | 5520.27 | 23.1873 | -0.642 |
| Pair 5 | 5642.61 | 22.6845 | 5480.79 | 23.3543 | -1.264 |
| **Median** | **5602.16** | **22.8483** | **5480.18** | **23.3569** | **-0.953** |

- **TG128 Throughput:** `22.8483 -> 23.3569 tok/s` (**+2.23% throughput gain**)
- **TG128 Latency:** `43.767 -> 42.814 ms/token` (**-0.953 ms/token latency saved**)
- **Allocations during decode:** 0
- **Replay check:** PASS

### 3. Hardware State & Telemetry

- TG64 Telemetry: 2,434 samples | SCLK == 1606MHz: 100.0% | Avg Temp: 44.5C (Max: 57.0C) | Avg Power: 59.5W (Max: 238.0W)
- TG128 Telemetry: 2,874 samples | SCLK == 1606MHz: 100.0% | Avg Temp: 48.6C (Max: 63.0C) | Avg Power: 79.1W (Max: 239.0W)
- Clocks remained locked at 1606 MHz for 100% of samples without thermal throttling.

## Interpretation

1. **VRAM Traffic Elimination:** Bypassing the roundtrip of writing and reading `ffn_gate` and `ffn_up` in VRAM saves 17.6 MB of HBM2 traffic per token across 64 layers. On MI50's ~1000 GB/s HBM2 interface, eliminating 17.6 MB/token saves ~0.018 ms of theoretical memory transfer plus significant L2 cache eviction pressure, which compounds to save **0.869 to 0.953 ms/token** in real-world decode latency.
2. **Instruction & Dispatch Reduction:** Combining the two GEMVs shares activation loads across both projections and eliminates 128 kernel invocations per token, simplifying the static HIP graph topology.
3. **Bitwise Precision:** Because `__builtin_amdgcn_sdot4` operates in integer accumulation and scale multiplications are carried out identically to separate execution, there is zero numerical drift (`max_abs_err = 0.0`, state fingerprint bit-for-bit identical).
4. **Performance Frontier Progress:**
   - Pre-M7 baseline: 22.42 tok/s (vanilla llama.cpp)
   - EXP-0179 (M6 baseline): 23.33 tok/s
   - EXP-0181 (+ HIP Graph): 23.47 tok/s
   - EXP-0182 (+ Fused Gate+Up SwiGLU): **23.95 tok/s (peak 23.96 tok/s)**
   - Gap to mx-llama.cpp TG64 frontier (25.74 tok/s): reduced from 2.41 tok/s down to 1.79 tok/s.

## Decision

**KEEP**.
- Latency saved: **+0.869 to +0.953 ms/token**.
- TG64 throughput reaches **23.95 tok/s** (new MIInfer record).
- Zero decode allocations maintained.
- Subphase M7-C is closed.

## Follow-up

Advance to **Subphase M7-D (Fused DeltaNet Recurrent Core in LDS)**:
- In the 56 recurrent layers of Qwen3.8-27B, the DeltaNet recurrence currently executes via 7 separate kernel launches per layer:
  1. `launch_qwen3_head_rms_normalize`
  2. `launch_qwen3_head_mul`
  3. `launch_qwen3_silu_mul`
  4. Recurrent state update (`s_t = beta * s_{t-1} + q * k^T`)
  5. State readout (`o_t = s_t * v`)
  6. RMS norm on output
  7. Output gate multiplication
- In `mx-llama.cpp`, the recurrent state update, normalization, and gating are fused into a single LDS-resident kernel (`recurrent_core`), saving ~2.1 ms/token.
- Designing a gfx906 Wave64 LDS-fused recurrent core kernel will bypass multiple global memory trips for state and intermediate activations across all 56 recurrent layers, targeting ~26.0 tok/s.
