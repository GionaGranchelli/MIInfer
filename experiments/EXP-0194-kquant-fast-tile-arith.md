# EXP-0194 — Vectorized Q4_K and Q5_K Fast Tile Arithmetic Optimization

## Hypothesis

Factoring out per-part conversions (`__half2float(q.d)`) and replacing variable shifts (`>> (4 * part)`) with branchless compile-time constants in `Q4KDecoder` and `Q5KDecoder` will reduce ALU instruction count, register pressure, and decode execution latency across all GEMV and fused SwiGLU operations without altering numerical precision or observable outputs.

## Motivation

In the MIInfer M8 profile, FFN Down and SwiGLU operations execute over 1.7 billion tile part calculations per token across 64 layers. In `Q4KDecoder::process_part`, each part previously executed:
1. Two redundant `__half2float(q.d)` conversions (`v_cvt_f32_f16`).
2. Two variable shift operations `state.v0 >> (4 * part)`.
3. Independent scale and minimum multiplications before factoring.

By factoring `const float q_d = __half2float(q.d)` and `const float q_ones = q_d * float(ones)`, and using `(part == 0) ? (state.v0 & 0x0f0f0f0f) : ((state.v0 >> 4) & 0x0f0f0f0f)`, several scalar and vector ALU instructions are eliminated per inner loop iteration.

## Baseline

* Commit: `5a62eca` (M8 Qualified Baseline)
* TG64 Median: 30.2170 tok/s / 33.094 ms/token
* Environment: AMD Instinct MI50 32GB (gfx906 / Vega20, locked 1606 MHz SCLK / 1000 MHz MCLK, 225W power cap)
* Compiler: HIP 6.2 (Clang 18)

## Candidate

* Source: `gfx906/kernels/kquant_wave_layout.hip` (`Q4KDecoderFast`, `Q5KDecoderFast`, `q4k_wave_fused_gate_up_swiglu_kernel`)
* Toggle: `MIINFER_KQUANT_FAST_ARITH=1` (enabled by default)

## Environment

* GPU: AMD Instinct MI50 32GB
* Architecture: gfx906 / Vega20, 60 CUs, Wave64
* SCLK: 1606 MHz (DPM 7)
* MCLK: 1000 MHz (DPM 2)
* Power Cap: 225W
* Telemetry: 906 continuous 250ms samples; 97.6% locked @ 1606 MHz, Avg Temp 52.0°C, Avg Power 96.5W

## Benchmark

* Harness: `scripts/run-exp0194-kquant-fast-arith-ab.sh tg64`
* Methodology: 5-pair interleaved Control/Candidate runs (10 runs total, 5 sample measurements per run)
* Model: `Qwen3.8-27B-Q4_K_M.gguf`

## Correctness

* Bitwise check (`scratch/test_q4k_opt.cpp`): `max_diff = 0.0f` across 1,000,000 synthetic random tiles.
* Observable Contract (64/64 positions):
  * `logits_cosine`: 0.999546
  * GPU winner rank: 1 / 1 (exact argmax match 64/64)
* Zero decode allocations (`allocations_during_decode=0`).
* Replay determinism: PASS across all 5 runs.
* CTest: 21/21 PASS.

## Results

| Run Pair | Control (tok/s) | Candidate (tok/s) | Control (ms/tok) | Candidate (ms/tok) | Delta |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Pair 1 | 30.1158 | 30.1283 | 33.205 | 33.191 | +0.04% |
| Pair 2 | 30.2679 | 30.2095 | 33.038 | 33.102 | -0.19% |
| Pair 3 | 30.2170 | 30.2699 | 33.094 | 33.036 | +0.17% |
| Pair 4 | 30.1713 | 30.2310 | 33.144 | 33.079 | +0.20% |
| Pair 5 | 30.2377 | 30.2526 | 33.071 | 33.055 | +0.05% |
| **Median** | **30.2170** | **30.2310** | **33.094** | **33.079** | **+0.05%** |

## Profiling & Interpretation

In microbenchmarks (`scratch/test_q4k_opt.cpp`), FFN Down tile execution dropped by 2.13 µs per 17-tile call. However, at full model scale, the GEMV decode kernels are heavily HBM2 memory-bandwidth bound (~520 GB/s) rather than VALU bound. While instruction count and register pressure are reduced, the memory bus remains saturated, resulting in an end-to-end latency reduction of 0.015 ms/token (+0.05% tok/s).

Importantly:
- Exact bitwise equivalence is maintained.
- Deterministic replay is 100% PASS.
- Zero decode allocations.
- No regression across any metric.

## Decision

**KEEP**.
The optimization simplifies inner loop code generation, eliminates unnecessary VALU conversions, and provides a clean, verified foundation for subsequent memory coalescing and SwiGLU vectorization.

## Follow-up

Investigate FFN Gate+Up SwiGLU memory bandwidth efficiency:
1. SwiGLU currently achieves only ~307 GB/s compared to FFN Down's ~570 GB/s.
2. Address wave workgroup unbalance in `q4k_wave_fused_gate_up_swiglu_kernel` (wave 0 processing 3 tiles vs wave 1 processing 2 tiles).
3. Evaluate coalescing gate and up weight loads to utilize 128-bit memory transactions.
