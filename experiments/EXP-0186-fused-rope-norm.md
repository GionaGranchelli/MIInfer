# EXP-0186 — Fused RoPE + Head Norm in Full Attention Layers

## Hypothesis

Fusing Q-split, RMS normalization, scale weighting, and RoPE section rotation into a single launch (`launch_qwen35_fused_q_split_norm_rope`), and fusing K RMS normalization, scale weighting, RoPE section rotation, and persistent KV cache storing into a single launch (`launch_qwen35_fused_k_norm_rope_kv_store`), will eliminate 6 kernel dispatches (96 static HIP graph nodes across 16 full-attention layers) and eliminate VRAM intermediate write/read round-trips for `query`, `query_norm`, `key_norm`, and `key_rope`, recovering measurable decode latency and advancing TG64 toward the M7 primary gate (27.24 tok/s).

## Motivation

In the Qwen3.8-27B architecture, 16 of the 64 layers are full multi-head attention layers. In the EXP-0185 baseline, each of these layers executed:
1. Stage 2:
   - `launch_qwen35_split_q_gate`: splits `qfull` into `query` and `gate`.
   - `launch_qwen3_head_rms_normalize`: computes per-head RMS norm of `query`.
   - `launch_qwen3_head_mul`: multiplies normalized query by learned weights.
2. Stage 4:
   - `launch_qwen3_head_rms_normalize`: computes per-head RMS norm of `key`.
   - `launch_qwen3_head_mul`: multiplies normalized key by learned weights.
3. Stage 6:
   - `launch_qwen35_rope_sections`: rotates `query_norm` into `query_rope`.
   - `launch_qwen35_rope_sections`: rotates `key_norm` into `key_rope`.
   - `launch_qwen3_kv_cache_store`: stores `key_rope` and `value` into persistent KV cache.

This pipeline suffered from:
- 8 separate kernel launches per attention layer (128 launches per token across 16 layers).
- 4 intermediate global memory round-trips (`query` 24 KiB, `query_norm` 24 KiB, `key_norm` 4 KiB, `key_rope` 4 KiB).
- Inefficient thread occupancy on small 4-head K projections.

By designing fused kernels:
- `qwen35_fused_q_split_norm_rope_kernel`:
  - 24 workgroups (one per query head) of 256 threads (`head_dim = 256`).
  - Reads `qfull`, writes `gate` to global memory, computes RMS norm in LDS, applies weight scale and RoPE section rotation in LDS/registers, and writes final `query_rope` directly to global memory.
- `qwen35_fused_k_norm_rope_kv_store_kernel`:
  - 4 workgroups (one per key head) of 256 threads.
  - Reads `key`, computes RMS norm in LDS, applies weight scale and RoPE in LDS/registers, and writes both rotated key and raw `value` directly to `key_cache` and `value_cache` at `[head][position][d]`.
- Replaces 8 kernel launches per layer with just 2 launches, eliminating 96 kernel nodes per token from the static HIP graph.

## Baseline

- Commit: `98ca0a2` (EXP-0185 Tiled Online-Softmax Attention with Gate Sigmoid Fusion).
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
  - `MIINFER_FAST_ARGMAX=1`
  - `MIINFER_TILED_ONLINE_ATTENTION=1`
- `MIINFER_FUSED_ROPE_NORM=0` (separate split, norm, head_mul, rope, and store kernels).

## Candidate

- Fused Q/K norm and RoPE kernels enabled via `MIINFER_FUSED_ROPE_NORM=1`.
- `qwen35_fused_q_split_norm_rope_kernel` and `qwen35_fused_k_norm_rope_kv_store_kernel` in `gfx906/kernels/qwen3_primitives.hip`.
- Integrated into `FullAttentionLayer::run` in `tools/m6a21_qwen35_gpu_hybrid_block.cpp`.

## Environment

- Hardware: AMD Instinct MI50 32GB HBM2 (gfx906 / Vega20, 60 CUs, Wave64)
- DPM Clock State: MANUAL SCLK level 7 (1606 MHz), MCLK level 2 (1000 MHz), 225W power cap
- ROCm version: 7.1.0
- Model: `Qwen3.8-27B-Q4_K_M.gguf`
- Fixture: `/tmp/m6a273-reference-p12`
- Telemetry: Continuous 250ms hardware sampling (`scripts/sample-gpu.sh`)

## Correctness

- Standalone verification (`scratch/test_fused_rope_norm.hip`):
  - Bit-for-bit exact zero error vs baseline across multiple positions (`diff_gate=0, diff_qrope=0, diff_kcache=0, diff_vcache=0`).
- 16-token autoregressive generation:
  - Tokens: `11,585,1044,264,5286,303,279,11759,314,76163,11,9903,26417,11,321,585` (exact match)
  - Replay check: PASS
  - Allocations during decode: 0
- Full CTest suite: 21 / 21 tests passed (100%).

## Benchmark Methodology

- 5 interleaved pairs (`Control 1 -> Candidate 1 -> ... -> Control 5 -> Candidate 5`) with 5s cooldown.
- Continuous 250ms hardware telemetry logging GPU clocks, temperature, and power.
- Evaluated on TG64 (64 tokens) and TG128 (128 tokens).
- Script: `scripts/run-exp0186-fused-rope-norm-ab.sh`.

## Results

### 1. TG64 (64 tokens)

| Pair | Control (ROPE_NORM=0) ms | Control tok/s | Candidate (ROPE_NORM=1) ms | Candidate tok/s | Latency Delta ms/tok |
|---|---|---|---|---|---|
| Pair 1 | 2404.99 | 26.6113 | 2398.17 | 26.6870 | -0.107 |
| Pair 2 | 2403.49 | 26.6280 | 2400.05 | 26.6661 | -0.054 |
| Pair 3 | 2407.56 | 26.5829 | 2395.96 | 26.7116 | -0.181 |
| Pair 4 | 2405.05 | 26.6106 | 2399.05 | 26.6772 | -0.094 |
| Pair 5 | 2402.83 | 26.6352 | 2398.49 | 26.6835 | -0.068 |
| **Median** | **2405.05** | **26.6106** | **2398.49** | **26.6835** | **-0.103** |

- **TG64 Throughput:** `26.6106 -> 26.6835 tok/s` (**+0.27% throughput gain**)
- **Peak TG64 Throughput:** **26.7116 tok/s**
- **TG64 Latency:** `37.579 -> 37.476 ms/token` (**-0.103 ms/token latency saved**)
- **Allocations during decode:** 0
- **Replay check:** PASS

### 2. TG128 (128 tokens)

| Pair | Control (ROPE_NORM=0) ms | Control tok/s | Candidate (ROPE_NORM=1) ms | Candidate tok/s | Latency Delta ms/tok |
|---|---|---|---|---|---|
| Pair 1 | 4823.43 | 26.5371 | 4814.24 | 26.5878 | -0.072 |
| Pair 2 | 4825.87 | 26.5237 | 4797.99 | 26.6779 | -0.218 |
| Pair 3 | 4833.97 | 26.4793 | 4807.36 | 26.6259 | -0.208 |
| Pair 4 | 4849.35 | 26.3953 | 4810.67 | 26.6075 | -0.302 |
| Pair 5 | 4831.54 | 26.4926 | 4801.29 | 26.6595 | -0.236 |
| **Median** | **4831.54** | **26.4926** | **4807.36** | **26.6259** | **-0.189** |

- **TG128 Throughput:** `26.4926 -> 26.6259 tok/s` (**+0.50% throughput gain**)
- **Peak TG128 Throughput:** **26.6779 tok/s**
- **TG128 Latency:** `37.746 -> 37.557 ms/token` (**-0.189 ms/token latency saved**)
- **Allocations during decode:** 0
- **Replay check:** PASS

### 3. Hardware State & Telemetry

- TG64 Telemetry: 2,381 samples | SCLK >= 1600MHz: 100.0% | Avg Temp: 42.5C (Max: 55.0C) | Avg Power: 58.6W (Max: 238.0W)
- TG128 Telemetry: 2,747 samples | SCLK >= 1600MHz: 100.0% | Avg Temp: 46.6C (Max: 62.0C) | Avg Power: 78.9W (Max: 243.0W)
- GPU locked at 1606 MHz for 100.0% of samples. Zero clock degradation or throttling.

## Interpretation

1. **Consistent Multi-Stage Latency Reduction**: Fusing the Q and K normalization, scale, RoPE, and KV store operations saved 0.103 ms/token on TG64 and 0.189 ms/token on TG128, while eliminating 96 static HIP graph nodes per token across the 16 full-attention layers.
2. **Current Standing vs M7 Success Gates**:
   - Primary Gate Target: **TG64 >= 27.24 tok/s (<= 36.71 ms/token)**.
   - Current MIInfer TG64: **26.68 tok/s (37.48 ms/token)** (peak 26.71 tok/s).
   - Remaining distance to Primary Gate: **0.766 ms/token** (~0.56 tok/s).
3. **Next Subphase (M7-H)**:
   - Residual addition and RMS normalization fusion (`launch_qwen3_fused_add_rms_norm` across all 64 layers) will eliminate 64 intermediate VRAM write/read round-trips and 64 kernel launches per token, targeting recovery of **~0.6–0.8 ms/token**, which will drive TG64 cleanly past the 27.24 tok/s Primary Gate!

## Decision

**KEEP**.
- Subphase M7-G is closed.
- Advance to Subphase M7-H: Inter-Layer Residual Addition + RMS Norm Fusion.
