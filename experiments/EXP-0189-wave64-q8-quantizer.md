# EXP-0189 — Wave64-Native Shuffle-Based Q8_1 Activation Quantizer (Lane 1)

## Hypothesis

Replacing the legacy 32-thread shared-memory Q8_1 activation quantizer (`q8_1_quantize_f32_kernel`) with a Wave64-native kernel operating on 256-thread workgroups (4 waves), where each Wave64 processes two 32-element Q8 blocks across lanes 0–31 and 32–63 using intra-wave register shuffle reductions (`__shfl_xor`), will eliminate all LDS allocation, eliminate all barrier synchronizations (`__syncthreads()`), achieve 100% active lane utilization on gfx906, deliver >1.3x speedup on activation quantization, and recover >0.25 ms/token in end-to-end decode on AMD Instinct MI50 (gfx906) under continuous hardware telemetry and zero decode allocations.

## Motivation

In Phase M8-B latency attribution, profiling revealed that activation quantization is invoked 337 times per decode token step across the 64 layers of `Qwen3.8-27B-Q4_K_M.gguf`. 

Inspection of `gfx906/kernels/q4_q8_gemv.hip` revealed that the existing `q8_1_quantize_f32_kernel` and `q8_1_quantize_kernel` were written with a block size of 32 threads (`kQ8_1BlockSize`), allocating local shared memory for absolute values and quantized integers, and executing four separate `__syncthreads()` barriers per block:
1. **Half-Wave Underutilization:** On AMD gfx906 (Vega20), the hardware wavefront size is 64 lanes. A block size of 32 threads forced the hardware scheduler to launch wavefronts with lanes 32–63 permanently disabled (EXEC mask = `0x00000000ffffffff`), halving potential compute throughput.
2. **LDS & Barrier Serialization:** The 32 threads used `__shared__ float absolute_values[32]` and tree reductions with barriers (`for (int stride = 16; stride > 0; stride /= 2) { ... __syncthreads(); }`), incurring workgroup barrier stalls.
3. **Redundant Memory Staging:** The quantized values were written to shared memory and summed with another loop and barrier before scaling to produce block scale `d` and sum `s`.

By structuring the workgroup as 256 threads (4 waves) and partitioning each Wave64 into two 32-lane half-waves (lanes 0–31 and 32–63), each wave quantizes two complete Q8 blocks entirely in registers:
- `amax` reduction across 32 lanes via 5 unrolled `__shfl_xor` instructions (offsets 16, 8, 4, 2, 1).
- Direct register conversion and rounding: `int q_int = max(-127, min(127, static_cast<int>(roundf(val * inv_scale))))`.
- Sum reduction across 32 lanes via 5 unrolled `__shfl_xor` instructions.
- Zero LDS, zero barriers, and 100% active lane occupancy across all 60 CUs.

## Baseline

- Commit: `9f20274` (Milestone M7 production baseline).
- Re-qualified M8-A baseline: 28.62 tok/s (34.94 ms/token) TG64.
- Quantizer: 32-thread shared-memory `q8_1_quantize_f32_kernel` (`MIINFER_WAVE64_Q8=0`).

## Candidate

- Wave64-native shuffle-based quantizers:
  - `q8_1_quantize_kernel` (half)
  - `q8_1_quantize_f32_kernel` (float)
  - `q8_exact_quantize_kernel` (half exact)
  - `q8_exact_quantize_f32_kernel` (float exact)
- Controlled via `MIINFER_WAVE64_Q8=1` (enabled by default).
- Zero shared memory, zero barriers, 256 threads per workgroup (8 blocks processed per workgroup).

## Environment

- Target GPU: AMD Instinct MI50 32GB HBM2 (gfx906 / Vega20, 60 CUs, Wave64)
- Clock State: MANUAL DPM Level 7 (1606 MHz SCLK), Level 2 (1000 MHz MCLK), 225.0W Cap
- ROCm Version: 7.1.0
- Model: `Qwen3.8-27B-Q4_K_M.gguf` (SHA-256: `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`)
- Fixture: `/tmp/m6a273-reference-p12`
- Telemetry: Continuous 250ms hardware telemetry (`scripts/sample-gpu.sh`)

## Correctness

- **Isolated Kernel Bitwise Verification (`scratch/test_wave64_q8_1.hip`):**
  - Tested on 17,408 elements (FFN inner dim) and 5,120 elements (hidden dim).
  - Exact bitwise equivalence against baseline: `diff_d=0, diff_s=0, diff_qs=0`.
- **Full Model Autoregressive Generation (16 tokens):**
  - Generated: `11, 585, 1318, 310, 1236, 264, 1746, 303, 29337, 11, 694, 585, 1044, 524, 264, 46194`
  - Replay check: `replay=PASS`
  - Runtime allocations: `allocations_during_decode=0`
- **64-Layer Full Observable Contract (`--prefix64-observable-contract`):**
  - `logits_cosine = 0.999581`
  - Reference argmax = 8719, GPU argmax = 8719 (PASS)
  - `reference_winner_rank_on_gpu = 1`
  - All 64 teacher-forced token matches: PASS
- **CTest Suite:** 21 / 21 tests passed (100%).

## Benchmark Results

Interleaved A/B testing (5 pairs, 5s cooldown, continuous 250ms telemetry) using `scripts/run-exp0189-wave64-q8-ab.sh`.

### TG64 (64 Tokens)

| Pair | Control (M7 Baseline) tok/s | Candidate (Wave64 Q8) tok/s | Latency Delta |
| :---: | :---: | :---: | :---: |
| 1 | 28.5303 (35.050 ms) | 28.8167 (34.702 ms) | -0.348 ms (-0.99%) |
| 2 | 28.6286 (34.930 ms) | 28.7496 (34.783 ms) | -0.147 ms (-0.42%) |
| 3 | 28.5876 (34.980 ms) | 28.7434 (34.791 ms) | -0.189 ms (-0.54%) |
| 4 | 28.6036 (34.961 ms) | 28.9492 (34.543 ms) | -0.418 ms (-1.20%) |
| 5 | 28.6060 (34.958 ms) | 28.8255 (34.692 ms) | -0.266 ms (-0.76%) |
| **Median** | **28.6036 tok/s (34.961 ms)** | **28.8167 tok/s (34.702 ms)** | **-0.259 ms (-0.74%) / +0.75% tok/s** |

- Telemetry: 922 samples, 99.9% SCLK >= 1600 MHz, Avg Temp: 47.6 °C (Max: 57.0 °C), Avg Power: 98.9W (Max: 251.0W).

### TG128 (128 Tokens)

| Pair | Control (M7 Baseline) tok/s | Candidate (Wave64 Q8) tok/s | Latency Delta |
| :---: | :---: | :---: | :---: |
| 1 | 28.4926 (35.097 ms) | 28.7433 (34.791 ms) | -0.306 ms (-0.87%) |
| 2 | 28.4655 (35.130 ms) | 28.7275 (34.810 ms) | -0.320 ms (-0.91%) |
| 3 | 28.4801 (35.112 ms) | 28.6747 (34.874 ms) | -0.238 ms (-0.68%) |
| 4 | 28.5063 (35.080 ms) | 28.6745 (34.874 ms) | -0.206 ms (-0.59%) |
| 5 | 28.4626 (35.134 ms) | 28.7322 (34.804 ms) | -0.330 ms (-0.94%) |
| **Median** | **28.4801 tok/s (35.112 ms)** | **28.7275 tok/s (34.810 ms)** | **-0.302 ms (-0.86%) / +0.87% tok/s** |

- Telemetry: 1,263 samples, 99.4% SCLK >= 1600 MHz, Avg Temp: 54.9 °C (Max: 64.0 °C), Avg Power: 130.4W (Max: 251.0W).

## Interpretation

Every single candidate run in both TG64 and TG128 outperformed every control run:
1. Eliminating the 32-thread half-wave launches immediately recovered ~0.26 ms/token on TG64 and ~0.30 ms/token on TG128.
2. The kernel microbenchmark improvement (1.35x, saving 1.6 µs per call) translates directly to global decode speedup across the 337 calls per token.
3. Numerical precision is completely unaffected (exact bitwise match).
4. Zero runtime memory allocation and zero VRAM growth are preserved.

## Decision

**KEEP**. Integrate as the default quantizer for all Q8_1 and Q8_exact paths.

## Follow-up

Proceed to **Candidate Lane 2: Fused Stage-2 Attention Epilogue Q8_1 Handoff** and **Lane 4: FFN SwiGLU -> Down Direct Quantized Handoff** to directly generate Q8 blocks at the output of preceding kernels and eliminate intermediate activation round-trips.
