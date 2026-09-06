# EXP-0184 — Fast 2-Stage Parallel Argmax Reduction

## Hypothesis

Replacing the single-workgroup serial argmax scan (`launch_qwen3_argmax` running on 1 thread block on 1 CU) with a 2-stage parallel reduction across 256 workgroups (utilizing all 60 CUs on MI50) will drop the argmax kernel duration from ~0.175–0.468 ms down to <10 µs, recovering ~0.4 to 0.5 ms/token decode latency while preserving bitwise numerical equivalence and tie-breaking order.

## Motivation

In the baseline implementation, `launch_qwen3_argmax` dispatched `dim3(1), dim3(256)`:
- Only 1 of MI50's 60 CUs was active; 59 CUs remained idle.
- Each thread in the block serially processed $(151936 + 255) / 256 \approx 594$ float logits in a serial loop.
- In HIP graph replay, this single serial kernel serialized execution at the end of every token decode, taking between 175 µs and 468 µs.

By refactoring `launch_qwen3_argmax` into a 2-stage hierarchical reduction:
1. **Stage 1 (`qwen3_argmax_stage1_kernel`)**:
   - Launches 256 thread blocks of 256 threads (grid stride across 65,536 threads), fully occupying all 60 CUs.
   - Each thread performs only 2 to 3 strided comparisons.
   - Each block performs an in-LDS tree reduction down to 1 partial winner (value + index) and writes to a module-level static device buffer (`s_argmax_partial_vals[256]` and `s_argmax_partial_idxs[256]`).
2. **Stage 2 (`qwen3_argmax_stage2_kernel`)**:
   - Launches 1 thread block of 256 threads to reduce the 256 partial winners into the final winning token index.
- Tie-breaking rules (`candidate_index < best_index` on tie) are strictly preserved at all stages.
- No allocations occur during decode (`s_argmax_partial_vals` and `s_argmax_partial_idxs` reside in static device BSS, 2 KiB total).
- Both stage kernels are captured seamlessly into the static HIP graph.

## Baseline

- Commit: `eb3ac7c` (EXP-0183 Fused DeltaNet Recurrent Core).
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
  - `MIINFER_FUSED_GATE_UP_SWIGLU=1`
  - `MIINFER_FUSED_RECURRENT_CORE=1`
- `MIINFER_FAST_ARGMAX=0` (single block serial argmax).

## Candidate

- 2-stage parallel argmax enabled via `MIINFER_FAST_ARGMAX=1`.
- `qwen3_argmax_stage1_kernel` and `qwen3_argmax_stage2_kernel` in `gfx906/kernels/qwen3_primitives.hip`.
- Automatically enabled for `elements > 256`.

## Environment

- Hardware: AMD Instinct MI50 32GB HBM2 (gfx906 / Vega20, 60 CUs, Wave64)
- DPM Clock State: MANUAL SCLK level 7 (1606 MHz), MCLK level 2 (1000 MHz), 225W power cap
- ROCm version: 7.1.0
- Model: `Qwen3.8-27B-Q4_K_M.gguf`
- Fixture: `/tmp/m6a273-reference-p12`
- Telemetry: Continuous 250ms hardware sampling (`scripts/sample-gpu.sh`)

## Correctness

- Standalone microbenchmark (`scratch/test_fast_argmax.hip` and `scratch/test_tie_break.hip`):
  - 151,936 random logits tested across multiple seeds: exact token match with baseline.
  - Tie-breaking verification: tested duplicate maximum values at arbitrary positions; confirmed strictly lowest index wins.
  - Execution time: dropped from 175.0 µs down to 7.8 µs (95.6% kernel latency reduction).
- 16-token autoregressive generation:
  - Tokens: `11,585,1044,264,5286,303,279,11759,314,76163,11,9903,26417,11,321,585` (exact match)
  - State fingerprint: `15952749137266575622` (exact match)
  - Replay check: PASS
  - Allocations during decode: 0
- 64-token autoregressive generation:
  - Tokens: 64/64 exact match (`first_token=11`, `last_token=28609`)
  - State fingerprint: `11163674888860358983` (exact match)
  - Replay check: PASS
  - Allocations during decode: 0
- Full CTest suite: 21 / 21 tests passed (100%), including updated `qwen3-primitives-test` with 151,936 elements and tie-break assertions.

## Benchmark Methodology

- 5 interleaved pairs (`Control 1 -> Candidate 1 -> ... -> Control 5 -> Candidate 5`) with 5s cooldown.
- Dedicated hardware telemetry sampler recording GPU clocks, temperature, and power at 250ms intervals.
- Tested across TG64 (64 tokens) and TG128 (128 tokens).
- Script: `scripts/run-exp0184-fast-argmax-ab.sh`.

## Results

### 1. TG64 (64 tokens)

| Pair | Control (ARGMAX=0) ms | Control tok/s | Candidate (ARGMAX=1) ms | Candidate tok/s | Latency Delta ms/tok |
|---|---|---|---|---|---|
| Pair 1 | 2500.16 | 25.5984 | 2468.10 | 25.9309 | -0.501 |
| Pair 2 | 2502.24 | 25.5771 | 2461.57 | 25.9997 | -0.635 |
| Pair 3 | 2493.77 | 25.6640 | 2472.62 | 25.8835 | -0.330 |
| Pair 4 | 2490.35 | 25.6992 | 2466.63 | 25.9463 | -0.371 |
| Pair 5 | 2507.04 | 25.5281 | 2465.07 | 25.9627 | -0.656 |
| **Median** | **2500.16** | **25.5984** | **2466.63** | **25.9463** | **-0.524** |

- **TG64 Throughput:** `25.5984 -> 25.9463 tok/s` (**+1.36% throughput gain**)
- **Peak TG64 Throughput:** **25.9997 tok/s** (surpasses `mx-llama.cpp` 25.74 tok/s frontier!)
- **TG64 Latency:** `39.065 -> 38.541 ms/token` (**-0.524 ms/token latency saved**)
- **Allocations during decode:** 0
- **Replay check:** PASS

### 2. TG128 (128 tokens)

| Pair | Control (ARGMAX=0) ms | Control tok/s | Candidate (ARGMAX=1) ms | Candidate tok/s | Latency Delta ms/tok |
|---|---|---|---|---|---|
| Pair 1 | 5125.24 | 24.9744 | 5086.76 | 25.1634 | -0.301 |
| Pair 2 | 5123.67 | 24.9821 | 5066.83 | 25.2624 | -0.444 |
| Pair 3 | 5140.59 | 24.8999 | 5067.88 | 25.2571 | -0.568 |
| Pair 4 | 5126.90 | 24.9664 | 5074.02 | 25.2266 | -0.413 |
| Pair 5 | 5130.84 | 24.9472 | 5076.58 | 25.2138 | -0.424 |
| **Median** | **5126.90** | **24.9664** | **5074.02** | **25.2266** | **-0.413** |

- **TG128 Throughput:** `24.9664 -> 25.2266 tok/s` (**+1.04% throughput gain**)
- **Peak TG128 Throughput:** **25.2624 tok/s**
- **TG128 Latency:** `40.054 -> 39.641 ms/token` (**-0.413 ms/token latency saved**)
- **Allocations during decode:** 0
- **Replay check:** PASS

### 3. Hardware State & Telemetry

- TG64 Telemetry: 2,369 samples | SCLK >= 1600MHz: 100.0% | Avg Temp: 43.6C (Max: 56.0C) | Avg Power: 58.7W (Max: 239.0W)
- TG128 Telemetry: 2,762 samples | SCLK >= 1600MHz: 100.0% | Avg Temp: 47.7C (Max: 63.0C) | Avg Power: 79.1W (Max: 241.0W)
- Clocks were locked at 1606 MHz for 100% of samples. Zero thermal throttling.

## Interpretation

1. **Clean Latency Recovery:** Recovered **+0.524 ms/token** on TG64 and **+0.413 ms/token** on TG128, directly consistent with the microbenchmark profiling (which showed 167 µs reduction in kernel compute time plus eliminated serialization stall on the command queue).
2. **Surpassing the Competitive Frontier on TG64:**
   - Prior benchmark competitor `mxxm-t/mx-llama.cpp` (b10904, commit `2e9d29fe`) achieved:
     - **TG64:** 25.74 tok/s (38.85 ms/token)
   - **MIInfer EXP-0184** achieves:
     - **TG64:** **25.95 tok/s (38.54 ms/token)**, with peak run of **26.00 tok/s**.
   - MIInfer has officially beaten `mx-llama.cpp` on TG64 throughput and latency!
3. **TG128 Standing:**
   - `mx-llama.cpp` TG128: 25.94 tok/s (38.55 ms/token).
   - MIInfer TG128: 25.23 tok/s (39.64 ms/token).
   - The remaining gap on TG128 (0.71 tok/s, ~1.09 ms/token) is primarily in the full-attention layers where context length grows from 1 to 128 tokens.

## Decision

**KEEP**.
- Subphase M7-E is closed.
- Latency saved: **+0.524 ms/token**.
- TG64 reaches **25.95 tok/s** (peak 26.00 tok/s).

## Follow-up

Advance to **Subphase M7-F (Tiled Online-Softmax Attention)**:
- In the 16 full-attention layers, as KV cache context grows from token 0 to token 128+, attention latency scales from 0.145 ms (at pos 63) to >0.35 ms (at pos 128+).
- Currently, attention uses non-tiled softmax reductions requiring separate global memory passes.
- Implementing tiled online-softmax attention (FlashAttention-style single-pass tile reduction in LDS) will keep attention kernel execution under 0.08 ms across all context lengths.
- This will recover ~1.5 to 2.0 ms/token across the 16 attention layers, targeting **>27.24 tok/s**, which will cross the **M7 Primary Success Gate**!
