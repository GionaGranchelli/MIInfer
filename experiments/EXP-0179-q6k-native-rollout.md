# EXP-0179 — Native Q6_K Wave64 Rollout & Activation Reuse

## Hypothesis

Replacing canonical GGUF Q6_K MMVQ across all Q6_K tensors in Qwen3.8-27B (`blk.*.attn_qkv.weight`, `blk.*.attn_v.weight`, and `blk.*.ffn_down.weight`) with gfx906-native `Q6KWaveTile` (two low nibble planes + two high 2-bit planes, 64-thread Wave64 execution), while simultaneously exploiting `Q8_1` activation reuse across projections (`attn_gate` reusing `attn_qkv`'s `q8_1`; `attn_v` reusing `attn_q`'s `q8_1`), will reduce per-token decode latency by >= 10 ms/token, beating the pinned llama.cpp baseline on AMD MI50 while maintaining zero decode allocations and exact numerical replay contract.

## Motivation

In EXP-0178, native Q5_K Wave64 rollout pushed MIInfer to 17.36 tok/s TG64 (57.62 ms/token).
However, a large fraction of the model's weights remained canonical Q6_K MMVQ:
- `attn_qkv.weight`: 24 of 48 recurrent layers are Q6_K (`[5120 x 10240]`)
- `attn_v.weight`: 9 of 17 full-attention layers are Q6_K (`[5120 x 1024]`)
- `ffn_down.weight`: 33 of 65 layers are Q6_K (`[17408 x 5120]`)

Canonical Q6_K MMVQ suffers from two major overheads on Vega20 / gfx906:
1. Canonical GGUF Q6_K layout requires complex scattered bit-unpacking and dot products with `Q8_K` activations, requiring an expensive separate `launch_q8_k_quantize_f32` pass.
2. In recurrent layers, `attn_qkv` and `attn_gate` both project `normalized`. With native Q6_K/Q4_K Wave64, `normalized` is quantized once to `Q8_1`, allowing `attn_gate` to completely eliminate its quantization dispatch.
3. In full-attention layers, `attn_q`, `attn_k`, and `attn_v` all project `normalized`. Native `Q6KWaveTile` consumes `Q8_1`, eliminating the `Q8_K` quantization pass previously required by canonical Q6_K `attn_v`.

## Baseline

- Commit: `ce0625a` (EXP-0178 complete with all Q4_K families + Q5_K SSM Out active).
- `MIINFER_Q4K_NATIVE_DOWN=1`
- `MIINFER_Q4K_NATIVE_GATE_UP=1`
- `MIINFER_Q4K_NATIVE_Q=1`
- `MIINFER_Q4K_NATIVE_ATTN_GATE=1`
- `MIINFER_Q4K_NATIVE_ATTN_OUT=1`
- `MIINFER_Q4K_NATIVE_K=1`
- `MIINFER_Q5K_NATIVE_SSM_OUT=1`
- `MIINFER_KQUANT_NATIVE_QKV=0`
- `MIINFER_KQUANT_NATIVE_V=0`
- `MIINFER_Q6K_NATIVE_DOWN=0`

## Candidate

- Native Q6_K Wave64 active across:
  - `MIINFER_KQUANT_NATIVE_QKV=1` (recurrent layers: 24 Q6_K + 24 Q4_K tensors, with `attn_gate` activation reuse)
  - `MIINFER_KQUANT_NATIVE_V=1` (full-attention layers: 9 Q6_K + 8 Q4_K tensors, with `attn_v` activation reuse)
  - `MIINFER_Q6K_NATIVE_DOWN=1` (FFN down: 33 Q6_K tensors)
- Packed into `Q6KWaveTile` layout via `pack_q6k_wave_tensor`.
- Canonical device buffer allocations completely bypassed when native is active.

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
  - `logits_cosine = 0.999588`
  - `top5_overlap = 4 / 5`
  - `poisoned_reset_replay = PASS`
  - `allocations_during_decode = 0`
- Unit tests: `miinfer-kquant-wave-test` PASS.
- CTest suite: 21 / 21 tests passed (100%).

## Benchmark Methodology

- Interleaved 5-pair A/B testing (`Control 1 -> Candidate 1 -> ... -> Control 5 -> Candidate 5`) with 5s cooldown.
- Full 250ms GPU hardware telemetry during the entire benchmark run.

## Results

### 1. TG64 (64 tokens)

| Run | Control (Q6K=0) ms | Control tok/s | Candidate (Q6K=1) ms | Candidate tok/s | Delta ms |
|---|---|---|---|---|---|
| Pair 1 | 3688.47 | 17.3514 | 2748.25 | 23.2875 | -940.22 |
| Pair 2 | 3687.11 | 17.3577 | 2746.51 | 23.3023 | -940.60 |
| Pair 3 | 3687.99 | 17.3536 | 2742.40 | 23.3372 | -945.59 |
| Pair 4 | 3680.59 | 17.3885 | 2740.43 | 23.3540 | -940.16 |
| Pair 5 | 3688.08 | 17.3532 | 2743.04 | 23.3318 | -945.04 |
| **Median** | **3687.99** | **17.3536** | **2743.04** | **23.3318** | **-944.95** |
| **Mean** | **3686.45** | **17.3609** | **2744.13** | **23.3226** | **-942.32** |

- **TG64 Throughput:** `17.3609 -> 23.3226 tok/s` (**+34.34% throughput**)
- **TG64 Latency:** `57.6007 -> 42.8770 ms/token` (**-14.7238 ms/token saved**)
- **Allocations during decode:** 0
- **Replay check:** PASS

### 2. TG128 (128 tokens)

| Run | Control (Q6K=0) ms | Control tok/s | Candidate (Q6K=1) ms | Candidate tok/s | Delta ms |
|---|---|---|---|---|---|
| Pair 1 | 7514.87 | 17.0329 | 5625.55 | 22.7533 | -1889.32 |
| Pair 2 | 7513.11 | 17.0369 | 5624.86 | 22.7561 | -1888.25 |
| Pair 3 | 7516.29 | 17.0297 | 5625.24 | 22.7546 | -1891.05 |
| Pair 4 | 7515.48 | 17.0315 | 5623.89 | 22.7601 | -1891.59 |
| Pair 5 | 7505.94 | 17.0532 | 5622.84 | 22.7643 | -1883.10 |
| **Median** | **7514.87** | **17.0329** | **5624.86** | **22.7561** | **-1890.01** |
| **Mean** | **7513.14** | **17.0368** | **5624.48** | **22.7577** | **-1888.66** |

- **TG128 Throughput:** `17.0368 -> 22.7577 tok/s` (**+33.58% throughput**)
- **TG128 Latency:** `58.6964 -> 43.9412 ms/token` (**-14.7552 ms/token saved**)
- **Allocations during decode:** 0
- **Replay check:** PASS

### 3. Comparison with Pinned llama.cpp Baseline

| Benchmark Metric | Pinned llama.cpp | MIInfer EXP-0178 | MIInfer EXP-0179 (This Exp) | Delta vs llama.cpp |
|---|---|---|---|---|
| **TG64 Throughput** | 22.42 tok/s | 17.36 tok/s | **23.32 tok/s** | **+4.01% faster** |
| **TG64 Latency** | 44.60 ms/token | 57.60 ms/token | **42.88 ms/token** | **-1.72 ms/tok lower** |
| **TG128 Throughput** | 22.50 tok/s | 17.04 tok/s | **22.76 tok/s** | **+1.16% faster** |
| **TG128 Latency** | 44.44 ms/token | 58.70 ms/token | **43.94 ms/token** | **-0.50 ms/tok lower** |

### 4. Hardware State & Telemetry

- TG64 Telemetry: 1976 samples, SCLK 1606 MHz: 100.0%, MCLK 1000 MHz: 100.0%, Max junction: 85.0 C.
- TG128 Telemetry: 2468 samples, SCLK 1606 MHz: 100.0%, MCLK 1000 MHz: 100.0%, Max junction: 93.0 C.
- Zero clock throttling, zero thermal throttling observed.

## Interpretation

1. Native Q6_K Wave64 execution combined with activation reuse delivers a massive performance breakthrough: **-14.72 ms/token saved** and **+34.34% throughput**.
2. MIInfer has officially beaten the pinned `llama.cpp` baseline on AMD MI50:
   - **TG64: 23.32 tok/s vs llama.cpp 22.42 tok/s (+4.0%)**
   - **TG128: 22.76 tok/s vs llama.cpp 22.50 tok/s (+1.2%)**
3. The Primary Success Gate (**TG64 >= 22.5 tok/s / <= 44.44 ms/token**) has been conclusively met under rigorous 5-pair interleaved benchmarking and continuous 250ms hardware state telemetry.

## Decision

**KEEP**.
- Latency recovered: **14.72 ms/token**.
- TG64 throughput reaches **23.32 tok/s** (surpassing 22.5 tok/s gate).
- TG128 throughput reaches **22.76 tok/s** (surpassing 22.5 tok/s gate).
- Zero decode allocations maintained.
- 64/64 deterministic replay match verified.

## Follow-up

Pursue the stretch goal (>= 25 tok/s / <= 40 ms/token):
1. **M7-B:** HIP graph capture for the static 64-layer decode trunk (eliminating ~400 kernel dispatches per token, recovering ~1.5–2.0 ms/token).
2. **M7-C:** DeltaNet recurrent path fusion (fusing norm, alpha/beta decay, gates, and recurrent state update in single LDS kernel, recovering ~3.0 ms/token).
3. **M7-D:** LM-head top-1 fused argmax (recovering ~1.0 ms/token).
