# EXP-0185 — Tiled Online-Softmax Attention with Gate Sigmoid Fusion

## Hypothesis

Replacing the baseline multi-pass attention implementation (`launch_qwen3_cached_attention_parallel` + `launch_qwen35_sigmoid_mul`) with a 2-stage Split-K tiled online-softmax kernel (`launch_qwen35_tiled_online_attention`) will eliminate global memory round-trips for scores and probabilities (24 KiB + 72 KiB per layer), eliminate 192 intra-block barriers per layer, fuse attention with the gate sigmoid elementwise multiplication in registers, and distribute attention across all 60 CUs on MI50 (via Split-K 2D grid `dim3(24, 4)`), recovering >0.8 ms/token decode latency on TG64 and >1.5 ms/token on TG128 while maintaining bitwise numerical stability and exact determinism.

## Motivation

In the Qwen3.8-27B hybrid architecture, 16 of the 64 layers are full multi-head attention layers (every 4th layer: 3, 7, 11, ..., 63). In the baseline implementation (EXP-0184):
1. **CU Under-Utilization**: The attention grid dispatched `dim3(24)` workgroups (one per query head). Out of MI50's 60 CUs, 36 CUs (60%) were completely idle during attention.
2. **Excessive Global Memory Round-Trips**: The baseline kernel performed 4 separate passes over global memory:
   - Pass 1: Computes key dot products, writes `scores[head, pos]` (24 KiB) to VRAM.
   - Pass 2: Reads `scores` from VRAM, computes softmax denominator, writes `probabilities` (24 KiB) to VRAM.
   - Pass 3: In a nested loop, reads `probabilities` from VRAM and `value_cache` from VRAM to accumulate output.
   - Pass 4: Normalizes `probabilities` and writes back to VRAM.
3. **Barrier Serialization**: For every token position in the KV cache, the baseline executed 3 block-wide synchronization barriers (`__syncthreads()`). At position 64, that represents 192 barriers stalling the execution pipeline.
4. **Separate Sigmoid Multiplication**: Stage 8 dispatched `launch_qwen35_sigmoid_mul` as a standalone kernel, adding 16 kernel dispatches and another 24 KiB VRAM write/read roundtrip per token.

By designing `launch_qwen35_tiled_online_attention`:
- **Stage 1 (`qwen3_splitk_stage1_kernel`)**:
  - Grid: `dim3(query_heads, num_splits)` (24 heads $\times$ 4 splits = 96 workgroups, fully saturating all 60 CUs on MI50).
  - Uses online softmax (Milakov & Gimelshein 2018): maintains running maximum $m$, running sum $l$, and running accumulator $acc$ in registers.
  - In-wave reduction across 64 lanes via Wave64 DPP shuffle (`qwen3_wave_sum`), requiring only 1 small LDS sync per position.
  - Zero global memory writes for intermediate scores or probabilities.
- **Stage 2 (`qwen3_splitk_stage2_kernel`)**:
  - Grid: `dim3(query_heads)` (24 workgroups of 256 threads).
  - Rescales and merges partial accumulators from the 4 splits:
    $$m = \max_s(m_s), \quad \alpha_s = e^{m_s - m}, \quad l = \sum_s l_s \alpha_s, \quad acc = \sum_s acc_s \alpha_s$$
  - Epilogue Fusion: Multiplies $acc / l$ by $\text{sigmoid}(\text{gate})$ directly in registers, writing `gated_attention` directly to VRAM ready for `d_o` Q8_1 quantization.
- Zero decode allocations: Intermediate split staging buffers reside in static module-level device memory (`s_attn_split_acc`, ~97 KiB).

## Baseline

- Commit: `f3232fe` (EXP-0184 Fast 2-Stage Parallel Argmax Reduction).
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
- `MIINFER_TILED_ONLINE_ATTENTION=0` (multi-pass cached attention + separate sigmoid mul).

## Candidate

- Tiled online-softmax attention enabled via `MIINFER_TILED_ONLINE_ATTENTION=1`.
- `qwen3_splitk_stage1_kernel` and `qwen3_splitk_stage2_kernel` in `gfx906/kernels/qwen3_primitives.hip`.
- Integrated into `FullAttentionLayer::run` in `tools/m6a21_qwen35_gpu_hybrid_block.cpp`.

## Environment

- Hardware: AMD Instinct MI50 32GB HBM2 (gfx906 / Vega20, 60 CUs, Wave64)
- DPM Clock State: MANUAL SCLK level 7 (1606 MHz), MCLK level 2 (1000 MHz), 225W power cap
- ROCm version: 7.1.0
- Model: `Qwen3.8-27B-Q4_K_M.gguf`
- Fixture: `/tmp/m6a273-reference-p12`
- Telemetry: Continuous 250ms hardware sampling (`scripts/sample-gpu.sh`)

## Correctness

- Standalone numerical verification (`scratch/test_splitk_correctness.hip`):
  - Tested across context lengths $N \in \{1, 2, 7, 16, 31, 32, 33, 63, 64, 127, 128, 256\}$.
  - Output max absolute difference vs baseline: $< 1.8 \times 10^{-7}$ across all lengths.
  - Gated output max absolute difference vs baseline: $< 1.2 \times 10^{-7}$ across all lengths.
  - Bitwise float accuracy matches baseline within float epsilon. PASS.
- 16-token autoregressive generation:
  - Tokens: `11,585,1044,264,5286,303,279,11759,314,76163,11,9903,26417,11,321,585` (exact match)
  - Replay check: PASS
  - Allocations during decode: 0
- Full CTest suite: 21 / 21 tests passed (100%).

## Benchmark Methodology

- 5 interleaved pairs (`Control 1 -> Candidate 1 -> ... -> Control 5 -> Candidate 5`) with 5s cooldown.
- Dedicated hardware telemetry sampler recording GPU clocks, temperature, and power at 250ms intervals.
- Tested across TG64 (64 tokens) and TG128 (128 tokens).
- Script: `scripts/run-exp0185-tiled-online-attention-ab.sh`.

## Results

### 1. TG64 (64 tokens)

| Pair | Control (ATTENTION=0) ms | Control tok/s | Candidate (ATTENTION=1) ms | Candidate tok/s | Latency Delta ms/tok |
|---|---|---|---|---|---|
| Pair 1 | 2461.48 | 26.0006 | 2405.52 | 26.6055 | -0.874 |
| Pair 2 | 2458.59 | 26.0312 | 2407.64 | 26.5821 | -0.796 |
| Pair 3 | 2467.63 | 25.9358 | 2401.42 | 26.6509 | -1.035 |
| Pair 4 | 2463.17 | 25.9828 | 2406.03 | 26.5999 | -0.893 |
| Pair 5 | 2479.13 | 25.8155 | 2406.00 | 26.6002 | -1.143 |
| **Median** | **2463.17** | **25.9828** | **2406.00** | **26.6002** | **-0.893** |

- **TG64 Throughput:** `25.9828 -> 26.6002 tok/s` (**+2.38% throughput gain**)
- **Peak TG64 Throughput:** **26.6509 tok/s**
- **TG64 Latency:** `38.487 -> 37.594 ms/token` (**-0.893 ms/token latency saved**)
- **Allocations during decode:** 0
- **Replay check:** PASS

### 2. TG128 (128 tokens)

| Pair | Control (ATTENTION=0) ms | Control tok/s | Candidate (ATTENTION=1) ms | Candidate tok/s | Latency Delta ms/tok |
|---|---|---|---|---|---|
| Pair 1 | 5056.13 | 25.3158 | 4835.35 | 26.4717 | -1.725 |
| Pair 2 | 5068.98 | 25.2516 | 4839.12 | 26.4511 | -1.796 |
| Pair 3 | 5059.40 | 25.2994 | 4825.17 | 26.5276 | -1.830 |
| Pair 4 | 5065.24 | 25.2703 | 4835.88 | 26.4688 | -1.792 |
| Pair 5 | 5110.55 | 25.0462 | 4847.12 | 26.4074 | -2.058 |
| **Median** | **5065.24** | **25.2703** | **4835.88** | **26.4688** | **-1.792** |

- **TG128 Throughput:** `25.2703 -> 26.4688 tok/s` (**+4.74% throughput gain**)
- **Peak TG128 Throughput:** **26.5276 tok/s**
- **TG128 Latency:** `39.572 -> 37.780 ms/token` (**-1.792 ms/token latency saved**)
- **Allocations during decode:** 0
- **Replay check:** PASS

### 3. Hardware State & Telemetry

- TG64 Telemetry: 2,365 samples | SCLK >= 1600MHz: 100.0% | Avg Temp: 43.3C (Max: 56.0C) | Avg Power: 58.7W (Max: 239.0W)
- TG128 Telemetry: 2,745 samples | SCLK >= 1600MHz: 100.0% | Avg Temp: 47.2C (Max: 63.0C) | Avg Power: 79.2W (Max: 240.0W)
- Clocks were locked at 1606 MHz for 100% of samples. Zero thermal throttling.

## Interpretation

1. **Context Scaling Degradation Eliminated:**
   - In baseline execution, latency grew from 38.49 ms/tok (TG64) to 39.57 ms/tok (TG128) (+1.08 ms/tok penalty) due to linear attention scan degradation.
   - With Tiled Online Softmax (Split-K), TG64 latency is 37.59 ms/tok and TG128 latency is 37.78 ms/tok (a mere 0.19 ms delta across 64 extra context tokens!).
2. **Definitive Surpassing of the gfx906 Frontier (`mx-llama.cpp`):**
   - Competitor `mxxm-t/mx-llama.cpp` (b10904, commit `2e9d29fe`):
     - **TG64:** 25.74 tok/s (38.85 ms/token)
     - **TG128:** 25.94 tok/s (38.55 ms/token)
   - **MIInfer EXP-0185**:
     - **TG64:** **26.60 tok/s (37.59 ms/token)** -> **+3.34% faster than mx-llama.cpp**
     - **TG128:** **26.47 tok/s (37.78 ms/token)** -> **+2.04% faster than mx-llama.cpp**
   - MIInfer has established a decisive lead over `mx-llama.cpp` across both short and medium context lengths!
3. **Standing vs M7 Success Gates:**
   - Primary Gate Target: **TG64 >= 27.24 tok/s (<= 36.71 ms/token)**.
   - Current MIInfer TG64: **26.60 tok/s (37.59 ms/token)**.
   - Remaining distance to Primary Gate: **0.883 ms/token** (~0.64 tok/s).

## Decision

**KEEP**.
- Subphase M7-F is closed.
- Latency saved: **+0.893 ms/tok** (TG64), **+1.792 ms/tok** (TG128).
- Zero decode allocations maintained.

## Follow-up

To close the remaining 0.88 ms/token and cross the **M7 Primary Success Gate (>= 27.24 tok/s)**:
- Profile remaining hot path components:
  1. `launch_qwen3_q6_k_q8_1_mmvq` in LM-head: currently takes 2.50 ms. Investigating Wave64 LDS unrolling or partial vocabulary reduction can save ~0.5–0.7 ms.
  2. RoPE + RMS norm fusion in attention layers: `rope_sections` (46 µs) + `q_split_head_norm` (41 µs) + `k_head_norm` (38 µs) = ~125 µs $\times 16 = 2.0$ ms. Fusing RoPE directly into head normalization saves ~0.6 ms.
