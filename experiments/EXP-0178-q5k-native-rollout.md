# EXP-0178 — Native Q5_K Wave64 Rollout (SSM Out Projections)

## Hypothesis

Replacing canonical GGUF Q5_K MMVQ for all 48 recurrent `blk.*.ssm_out.weight` [6144 x 5120] projections with gfx906-native `Q5KWaveTile` (896 bytes per 1024 weights, Wave64 two low nibble planes + high 5th-bit plane) will reduce per-token decode latency by >= 1.0 ms/token (~1.75x microbenchmark speedup), maintaining zero decode allocations and exact numerical replay contract.

## Motivation

In EXP-0177, the native `Q4KWaveTile` layout closed 41.4% of the decode gap to pinned llama.cpp (14.54 -> 17.02 tok/s TG64).
However, all 48 recurrent layers still used canonical Q5_K MMVQ for `ssm_out.weight` (6144 x 5120).
Microbenchmarking in `bench/kquant_layout_bench` showed canonical Q5_K MMVQ taking 64.10 us vs 36.50 us for native Wave64 (1.76x speedup, saving 27.6 us per layer). Across 48 recurrent layers, the analytical recoverable latency is ~1.32 ms/token.
Additionally, native Q5_K Wave64 layout is 896 bytes per 1024 weights (0.875 bytes/weight) compared to canonical Q5_K's 1152 bytes per 1024 weights (1.125 bytes/weight), yielding a 22.2% reduction in weight memory.

## Baseline

- Commit: `71fb4e8` (EXP-0177 complete with all Q4_K families active: Down, Gate/Up, Q, Attn Gate, Attn Out, Attn K).
- `MIINFER_Q4K_NATIVE_DOWN=1`
- `MIINFER_Q4K_NATIVE_GATE_UP=1`
- `MIINFER_Q4K_NATIVE_Q=1`
- `MIINFER_Q4K_NATIVE_ATTN_GATE=1`
- `MIINFER_Q4K_NATIVE_ATTN_OUT=1`
- `MIINFER_Q4K_NATIVE_K=1`
- `MIINFER_Q5K_NATIVE_SSM_OUT=0`

## Candidate

- Native Q5_K Wave64 active for all 48 recurrent layers:
  `MIINFER_Q5K_NATIVE_SSM_OUT=1`
- Weights packed into `Q5KWaveTile` layout via `pack_q5k_wave_tensor`.
- Canonical device buffer allocation completely bypassed when native is active.

## Environment

- Hardware: AMD Instinct MI50 32GB HBM2 (gfx906 / Vega20, 60 CUs, Wave64)
- DPM Clock State: MANUAL SCLK level 7 (1606 MHz), MCLK level 2 (1000 MHz)
- ROCm version: 7.1.0
- Model: `Qwen3.8-27B-Q4_K_M.gguf` (sha256: verified)
- Fixture: `/tmp/m6a273-reference-p12`
- Telemetry: Continuous 250ms hardware sampling (`scripts/sample-gpu.sh`)

## Correctness

- Teacher-Forced 64-step Contract (`--prefix64-observable-contract`):
  - 64 / 64 positions exact match (`reference_argmax == gpu_argmax` for all pos 0..63)
  - `logits_cosine = 0.999607`
  - `top5_overlap = 5 / 5`
  - `poisoned_reset_replay = PASS`
  - `allocations_during_decode = 0`
- Unit tests: `miinfer-kquant-wave-test` PASS (synthetic & real GGUF tensors).
- CTest suite: 21 / 21 tests passed (100%).

## Benchmark Methodology

- Interleaved 5-pair A/B testing (`Control 1 -> Candidate 1 -> ... -> Control 5 -> Candidate 5`) with 5s cooldown.
- Full 250ms GPU hardware telemetry during the entire benchmark run.

## Results

### 1. TG64 (64 tokens)

| Run | Control (SSM_OUT=0) ms | Control tok/s | Candidate (SSM_OUT=1) ms | Candidate tok/s | Delta ms |
|---|---|---|---|---|---|
| Pair 1 | 3764.98 | 16.9987 | 3688.22 | 17.3526 | -76.76 |
| Pair 2 | 3763.16 | 17.0070 | 3685.26 | 17.3665 | -77.90 |
| Pair 3 | 3765.52 | 16.9963 | 3687.70 | 17.3550 | -77.82 |
| Pair 4 | 3765.53 | 16.9963 | 3688.51 | 17.3512 | -77.02 |
| Pair 5 | 3763.45 | 17.0057 | 3680.84 | 17.3873 | -82.61 |
| **Median** | **3764.98** | **17.0012** | **3687.70** | **17.3550** | **-77.28** |

- **TG64 Throughput:** `17.0012 -> 17.3550 tok/s` (**+2.08%**)
- **TG64 Latency:** `58.828 -> 57.620 ms/token` (**-1.208 ms/token saved**)
- **Allocations during decode:** 0
- **Replay check:** PASS

### 2. TG128 (128 tokens)

| Run | Control (SSM_OUT=0) ms | Control tok/s | Candidate (SSM_OUT=1) ms | Candidate tok/s | Delta ms |
|---|---|---|---|---|---|
| Pair 1 | 7666.46 | 16.6961 | 7509.15 | 17.0459 | -157.31 |
| Pair 2 | 7671.41 | 16.6853 | 7506.83 | 17.0511 | -164.58 |
| Pair 3 | 7665.48 | 16.6982 | 7514.87 | 17.0329 | -150.61 |
| Pair 4 | 7653.97 | 16.7234 | 7510.58 | 17.0426 | -143.39 |
| Pair 5 | 7665.97 | 16.6972 | 7504.44 | 17.0566 | -161.53 |
| **Median** | **7665.97** | **16.6972** | **7509.15** | **17.0459** | **-156.82** |

- **TG128 Throughput:** `16.6972 -> 17.0459 tok/s` (**+2.09%**)
- **TG128 Latency:** `59.890 -> 58.665 ms/token` (**-1.225 ms/token saved**)
- **Allocations during decode:** 0
- **Replay check:** PASS

### 3. Hardware State & Telemetry

- TG64 Telemetry: 1506 samples, SCLK 1606 MHz: 100.0%, MCLK 1000 MHz: 100.0%, Max junction: 80.0 C.
- TG128 Telemetry: 2065 samples, SCLK 1606 MHz: 100.0%, MCLK 1000 MHz: 100.0%, Max junction: 87.0 C.
- Zero clock throttling or thermal throttling observed.

## Interpretation

The empirical saving (-1.208 ms/token on TG64 and -1.225 ms/token on TG128) closely matches the analytical expectation (-1.32 ms/token from 48 layers * 27.6 us).
Native Q5_K Wave64 breaks the 17 tok/s boundary on TG128 for the first time.
Zero regressions were observed across 64 teacher-forced verification steps.

## Decision

**KEEP**.
- Latency recovered: **1.21 ms/token**.
- TG64 throughput reaches **17.36 tok/s**.
- TG128 throughput reaches **17.05 tok/s**.
- Zero decode allocations maintained.

## Follow-up

Proceed to Candidate 2: Native Q6_K Wave64 rollout across:
1. `blk.*.attn_qkv.weight` (24 Q6_K + 24 Q4_K recurrent layers) with activation reuse in `attn_gate`.
2. `blk.*.attn_v.weight` (17 full-attention layers) with activation reuse of `q8_1`.
3. `blk.*.ffn_down.weight` (33 Q6_K layers).
