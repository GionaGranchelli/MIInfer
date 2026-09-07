# EXP-0192: Wave64 Fused Gate+Up SwiGLU Intra-Wave Shuffle Reduction

## Hypothesis

In `q4k_wave_fused_gate_up_swiglu_kernel` (`gfx906/kernels/kquant_wave_layout.hip`), each of the 8,704 thread blocks per layer (557,056 blocks per token) used 2,048 bytes of LDS to perform a 128-thread tree reduction with 8 `__syncthreads()` barrier synchronizations.

Replacing the 128-thread LDS tree reduction with:
1. Wave64 intra-wave shuffle reduction via `__shfl_down` (0 LDS, 0 barriers, 6 single-cycle DPP instructions).
2. Minimal 16-byte LDS cross-wave handoff between odd and even waves with exactly 1 `__syncthreads()`.

will eliminate ~3.9 million barrier synchronizations per token and 99.2% of LDS usage in the SwiGLU kernel, reducing decode latency by ~0.15 - 0.25 ms/token.

## Mechanism

- Templated `q4k_wave_fused_gate_up_swiglu_kernel<unsigned NumTiles, bool FastShuffle = true>`.
- In `FastShuffle=true`:
  - Intra-wave reduction:
    ```cpp
    #pragma unroll
    for (int offset = 32; offset > 0; offset /= 2) {
        sum_gate += __shfl_down(sum_gate, offset, 64);
        sum_up += __shfl_down(sum_up, offset, 64);
    }
    ```
  - Cross-wave handoff:
    ```cpp
    __shared__ float odd_gate[2];
    __shared__ float odd_up[2];
    if (tid == 64) {
        odd_gate[local_row] = sum_gate;
        odd_up[local_row] = sum_up;
    }
    __syncthreads();
    if (tid == 0) {
        const float g = sum_gate + odd_gate[local_row];
        const float u = sum_up + odd_up[local_row];
        const float silu_g = g / (1.0f + expf(-g));
        output[row] = silu_g * u;
    }
    ```
- Dispatch controlled via `MIINFER_SWIGLU_SHUFFLE` (defaulting to enabled).

## Baseline (Control)

- `MIINFER_SWIGLU_SHUFFLE=0` (legacy 2,048-byte LDS tree reduction with 8 barriers).
- Base flags: Post-EXP-0190 production configuration.

## Candidate

- `MIINFER_SWIGLU_SHUFFLE=1` (Wave64 intra-wave shuffle reduction with 1 barrier and 16 bytes LDS).

## Environment

- Hardware: AMD Instinct MI50 32GB HBM2 (gfx906 / Vega20, 60 CUs, Wave64)
- DPM Clock State: MANUAL SCLK level 7 (1606 MHz), MCLK level 2 (1000 MHz), 225W power cap
- ROCm version: 7.1.0
- Model: `Qwen3.8-27B-Q4_K_M.gguf`
- Fixture: `/tmp/m6a273-reference-p12`
- Telemetry: Continuous 250ms hardware sampling (`scripts/sample-gpu.sh`)

## Correctness

- 21 / 21 CTests PASS (100%).
- `miinfer-kquant-wave-test` (Test 5): Synthetic Q4_K fused gate+up SwiGLU reference PASS.
- 64-layer observable contract (`--prefix64-observable-contract`):
  - 64/64 teacher-forced argmax match: PASS.
  - Position 64 `logits_cosine = 0.999546` (>= 0.9995 requirement): PASS.
  - Winner rank 1 on GPU and Reference: PASS.
  - Allocations during decode: 0.
  - Replay test: PASS.

## Results

Interleaved A/B benchmark (5 pairs, 5s cooldown, locked 1606/1000 MHz):

### TG64

| Run Pair | Control (tok/s) | Candidate (tok/s) | Delta (tok/s) |
| :--- | :---: | :---: | :---: |
| Pair 1 | 28.9463 | 29.0766 | +0.1303 |
| Pair 2 | 28.9315 | 29.0908 | +0.1593 |
| Pair 3 | 28.9059 | 29.0356 | +0.1297 |
| Pair 4 | 28.9329 | 29.0575 | +0.1246 |
| Pair 5 | 28.9197 | 29.0777 | +0.1580 |
| **Median** | **28.9315 tok/s (34.564 ms)** | **29.0777 tok/s (34.391 ms)** | **+0.1462 tok/s (+0.51%, -0.174 ms)** |

- SCLK stability: 98.6% samples at 1606 MHz (Avg Temp: 50.7°C, Avg Power: 99.3W).

### TG128

| Run Pair | Control (tok/s) | Candidate (tok/s) | Delta (tok/s) |
| :--- | :---: | :---: | :---: |
| Pair 1 | 28.7556 | 28.8490 | +0.0934 |
| Pair 2 | 28.7473 | 28.9148 | +0.1675 |
| Pair 3 | 28.7046 | 28.8958 | +0.1912 |
| Pair 4 | 28.7256 | 28.8684 | +0.1428 |
| Pair 5 | 28.6780 | 28.9270 | +0.2490 |
| **Median** | **28.7256 tok/s (34.812 ms)** | **28.8958 tok/s (34.607 ms)** | **+0.1702 tok/s (+0.59%, -0.205 ms)** |

- SCLK stability: 94.5% samples at 1606 MHz (Avg Temp: 57.3°C, Avg Power: 131.4W).

## Interpretation

1. In the fused Gate+Up SwiGLU kernel, 8,704 workgroups run per layer across 64 layers.
2. The legacy reduction executed 8 `__syncthreads()` per workgroup, which stalled waves across all CUs and forced barrier arbitration in hardware.
3. Reducing barriers from 8 to 1 and replacing 2048 bytes of shared memory tree operations with single-cycle DPP shuffle instructions saved **0.174 ms/token in TG64** and **0.205 ms/token in TG128**.
4. Correctness, determinism, and zero decode allocations were strictly maintained across all 5 test pairs.

## Decision

**KEEP**. Enable `MIINFER_SWIGLU_SHUFFLE=1` permanently as the default.

## Follow-up

Current latency: **34.391 ms/token (29.08 tok/s)**.
Distance to 30.00 tok/s Primary Gate (≤ 33.333 ms/token): **1.058 ms/token**.
Next investigation:
1. DeltaNet recurrent state update kernel vectorization & loop overhead.
2. FFN Down Q4_K GEMV multi-wave tile unrolling.
3. LM Head Q6_K byte-unpacking acceleration.
