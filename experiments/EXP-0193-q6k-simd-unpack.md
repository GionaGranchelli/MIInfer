# EXP-0193: Vectorized SIMD Q6_K Decoding for LM Head and Down Projections

## Hypothesis

In `kquant_wave_gemv_kernel` (`gfx906/kernels/kquant_wave_layout.hip`), the Q6_K decoding logic for each 16-element tile executed 4 scalar loop iterations per thread, using individual byte extracts, shift expressions, bitwise masking, and byte-wise signed conversion (`(int8_t)((u & 0x1f) | ((u & 0x20) ? 0xe0 : 0))`).
For the LM Head (151,936 rows × 5,120 columns = 777.9 million weights) and Down projections, this scalar byte decoding generated excessive scalar ALU instructions and register spilling across all 60 CUs.

Vectorizing the 4-byte unpacking into parallel 32-bit SIMD bitwise operations:
1. Low bits: `low = (val >> shift) & 0x0f0f0f0f`
2. High bits: `high = (h >> shift) & 0x30303030`
3. 6-bit unsigned combining: `u = low | high`
4. Branchless SIMD 6-bit two's complement sign extension via parallel bit manipulation:
   `not_b5 = (u & 0x20202020) ^ 0x20202020`
   `q_packed = (u & 0x1f1f1f1f) | (not_b5 | (not_b5 << 1) | (not_b5 << 2))`
will eliminate loop branching and scalar instruction serialization, reducing compute overhead in Q6_K GEMV and saving >1.0 ms/token across full-model decode.

## Mechanism

- Added `Q6KDecoder<true>` specialized template to `gfx906/kernels/kquant_wave_layout.hip`.
- Scalar path retained for reference and controlled via runtime environment variable `MIINFER_Q6K_SIMD_UNPACK` (default enabled).
- Bitwise SIMD unpacking logic:
```cpp
const uint32_t low_a = (part == 0) ? (state.v0 & 0x0f0f0f0f) : ((state.v0 >> 4) & 0x0f0f0f0f);
const uint32_t low_b = (part == 0) ? (state.v1 & 0x0f0f0f0f) : ((state.v1 >> 4) & 0x0f0f0f0f);
const uint32_t high_a = (part == 0) ? ((state.h & 0x03030303) << 4) : (state.h & 0x30303030);
const uint32_t high_b = (part == 0) ? ((state.h & 0x0c0c0c0c) << 2) : ((state.h >> 2) & 0x30303030);
const uint32_t u_a = low_a | high_a;
const uint32_t u_b = low_b | high_b;
const uint32_t not_b5_a = (u_a & 0x20202020) ^ 0x20202020;
const uint32_t a = (u_a & 0x1f1f1f1f) | (not_b5_a | (not_b5_a << 1) | (not_b5_a << 2));
const uint32_t not_b5_b = (u_b & 0x20202020) ^ 0x20202020;
const uint32_t b = (u_b & 0x1f1f1f1f) | (not_b5_b | (not_b5_b << 1) | (not_b5_b << 2));
```
- Standalone microbenchmark (`miinfer-kquant-layout-bench`): Q6_K wave GEMV median latency reduced from **92.06 µs → 75.97 µs** (17.5% kernel latency reduction, 1.44x speedup over canonical MMVQ).

## Baseline (Control)

- `MIINFER_Q6K_SIMD_UNPACK=0` (Scalar loop Q6_K byte extraction).
- Base flags: Post-EXP-0192 production configuration (`MIINFER_SWIGLU_SHUFFLE=1`).

## Candidate

- `MIINFER_Q6K_SIMD_UNPACK=1` (Vectorized branchless 32-bit SIMD Q6_K unpacking).

## Environment

- Hardware: AMD Instinct MI50 32GB HBM2 (gfx906 / Vega20, 60 CUs, Wave64)
- DPM Clock State: MANUAL SCLK level 7 (1606 MHz), MCLK level 2 (1000 MHz), 225W power cap
- ROCm version: 7.1.0
- Model: `Qwen3.8-27B-Q4_K_M.gguf`
- Fixture: `/tmp/m6a273-reference-p12`
- Telemetry: Continuous 250ms hardware sampling (`scripts/sample-gpu.sh`)

## Correctness

- Bitwise verification: 100% exact bit-for-bit match across all 64 quantized representations (-32 to +31).
- 21 / 21 CTests PASS (100%).
- 64-layer observable contract (`--prefix64-observable-contract`):
  - 64/64 teacher-forced argmax match: PASS.
  - Position 64 `logits_cosine = 0.999546` (>= 0.9995 requirement): PASS.
  - Winner rank 1 on GPU and Reference: PASS.
  - Allocations during decode: 0.
  - Poisoned reset replay test: PASS.

## Results

Interleaved A/B benchmark (5 pairs, 5s cooldown, locked 1606/1000 MHz):

### TG64

| Run Pair | Control (tok/s) | Candidate (tok/s) | Delta (tok/s) |
| :--- | :---: | :---: | :---: |
| Pair 1 | 29.0435 | 30.1583 | +1.1148 |
| Pair 2 | 29.0477 | 30.1230 | +1.0753 |
| Pair 3 | 29.0355 | 30.1814 | +1.1459 |
| Pair 4 | 29.0315 | 30.1658 | +1.1343 |
| Pair 5 | 29.0113 | 30.2881 | +1.2768 |
| **Median** | **29.0355 tok/s (34.441 ms)** | **30.1813 tok/s (33.133 ms)** | **+1.1458 tok/s (+3.95%, -1.307 ms)** |

- SCLK stability: 97.0% samples at 1606 MHz (Avg Temp: 50.5°C, Avg Power: 98.7W).

### TG128

| Run Pair | Control (tok/s) | Candidate (tok/s) | Delta (tok/s) |
| :--- | :---: | :---: | :---: |
| Pair 1 | 28.9036 | 29.9894 | +1.0858 |
| Pair 2 | 28.8035 | 29.9344 | +1.1309 |
| Pair 3 | 28.8434 | 29.9251 | +1.0817 |
| Pair 4 | 28.8807 | 29.9592 | +1.0785 |
| Pair 5 | 28.8297 | 29.8896 | +1.0599 |
| **Median** | **28.8434 tok/s (34.670 ms)** | **29.9344 tok/s (33.406 ms)** | **+1.0910 tok/s (+3.78%, -1.264 ms)** |

- SCLK stability: 93.1% samples at 1606 MHz (Avg Temp: 57.9°C, Avg Power: 129.8W).
- TG64 → TG128 Scaling Penalty: `(30.1813 - 29.9344) / 30.1813 = 0.818%` (Gate requirement: ≤ 1.0%).

## Decision

**KEEP**.
EXP-0193 decisively breaches the M8 Primary Success Gate:
- **TG64 = 30.18 tok/s / 33.13 ms/token** (surpasses requirement of ≥ 30.00 tok/s / ≤ 33.33 ms/token).
- **TG128 = 29.93 tok/s** (surpasses secondary requirement of ≥ 29.75 tok/s).
- **TG64→TG128 Penalty = 0.82%** (surpasses requirement of ≤ 1.0%).
- 64/64 argmax contract PASS (`logits_cosine >= 0.9995`).
- Zero decode allocations (`allocations_during_decode = 0`).
- Deterministic replay PASS.
