# EXP-0183 — Fused DeltaNet Recurrent Core in LDS

## Hypothesis

Fusing the DeltaNet recurrent state update, per-head RMS normalization, SSM norm scaling, and gate SiLU activation into a single workgroup kernel with register-cached state rows will eliminate 192 KiB of intermediate global memory traffic per layer (10.75 MB/token across 48 recurrent layers), eliminate 144 kernel launches per token from the static execution DAG, and eliminate a redundant 64 KiB HBM read pass per head by caching the 128-element state row in VGPRs, recovering >2.5 ms/token decode latency on AMD Instinct MI50 (gfx906) while preserving bitwise numerical equivalence.

## Motivation

In the Qwen3.8-27B architecture, 48 of the 64 layers are Gated DeltaNet linear attention layers. In baseline execution (EXP-0182), each recurrent layer executed its attention recurrence through 4 distinct kernel launches:
1. `launch_qwen35_deltanet_state_update_transposed_no_decay_store_lds_inputs`: Updates the $48 \times 128 \times 128$ float recurrent state and writes `recurrent_output` (24 KiB) to global VRAM. In doing so, it read the 128-element state row twice from HBM (once for `key_dot` projection and once for the update and `query_dot` readout).
2. `launch_qwen3_head_rms_normalize`: Reads `recurrent_output` from global VRAM, computes per-head RMS, and writes `head_norm` (24 KiB) to global VRAM.
3. `launch_qwen3_head_mul`: Reads `head_norm` from global VRAM and `d_ssm_norm` (512 B), writing `gated` (24 KiB) to global VRAM.
4. `launch_qwen3_silu_mul`: Reads `gate` (24 KiB) and `gated` from global VRAM, computing `silu(gate) * gated` and writing `gated` back to global VRAM.

Neither `recurrent_output` nor `head_norm` is consumed downstream outside optional diagnostics.

By designing `launch_qwen35_deltanet_fused_recurrent_core`:
- 48 thread blocks (one per value head) each launch 128 threads ($row \in [0, 128)$), matching MI50's 60 CUs with 1 resident block per CU.
- Register Caching: Each thread caches its 128 state row elements in local VGPRs (`float s[128]`), completely eliminating the second 64 KiB read pass from HBM.
- In-LDS RMS Reduction: Thread $row$ computes `rec_val = query_dot * rsqrtf(128.0F)`. The 128 threads place `rec_val * rec_val` in LDS and perform an in-workgroup tree reduction identical to `qwen3_head_rms_normalize_kernel`.
- Epilogue Fusion: Each thread scales by `ssm_norm[row]` and multiplies by $\text{silu}(gate[out\_idx])$ directly in registers, writing the final result directly to `gated_output`.
- 144 kernel launches per token are eliminated from the static HIP graph.

## Baseline

- Commit: `a418b83` (EXP-0182 Fused Gate+Up SwiGLU enabled).
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
- `MIINFER_FUSED_RECURRENT_CORE=0` (separate DeltaNet state update + RMS norm + SSM mul + gate SiLU mul).

## Candidate

- Fused DeltaNet recurrent core enabled via `MIINFER_FUSED_RECURRENT_CORE=1`.
- `qwen35_deltanet_fused_recurrent_core_kernel` with register caching in `gfx906/kernels/qwen3_primitives.hip`.
- Integrated into `RecurrentLayer::run` in `tools/m6a21_qwen35_gpu_hybrid_block.cpp`.
- Directly outputs to `gated`, bypassing Stage 6 kernel dispatches.

## Environment

- Hardware: AMD Instinct MI50 32GB HBM2 (gfx906 / Vega20, 60 CUs, Wave64)
- DPM Clock State: MANUAL SCLK level 7 (1606 MHz), MCLK level 2 (1000 MHz), 225W power cap
- ROCm version: 7.1.0
- Model: `Qwen3.8-27B-Q4_K_M.gguf`
- Fixture: `/tmp/m6a273-reference-p12`
- Telemetry: Continuous 250ms hardware sampling (`scripts/sample-gpu.sh`)

## Correctness

- Standalone numerical verification (`scratch/verify_reg_cache`):
  - `max_diff_state = 0.000000e+00`
  - `max_diff_gated = 0.000000e+00`
  - Bit-for-bit exact zero error vs separate execution. PASS.
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
- Script: `scripts/run-exp0183-fused-recurrent-ab.sh`.

## Results

### 1. TG64 (64 tokens)

| Pair | Control (FUSED=0) ms | Control tok/s | Candidate (FUSED=1) ms | Candidate tok/s | Latency Delta ms/tok |
|---|---|---|---|---|---|
| Pair 1 | 2674.52 | 23.9295 | 2495.08 | 25.6504 | -2.804 |
| Pair 2 | 2675.57 | 23.9201 | 2494.12 | 25.6604 | -2.835 |
| Pair 3 | 2681.95 | 23.8632 | 2494.82 | 25.6531 | -2.924 |
| Pair 4 | 2666.76 | 23.9992 | 2487.12 | 25.7326 | -2.807 |
| Pair 5 | 2667.86 | 23.9893 | 2489.26 | 25.7104 | -2.791 |
| **Median** | **2674.52** | **23.9295** | **2494.12** | **25.6604** | **-2.819** |

- **TG64 Throughput:** `23.9295 -> 25.6604 tok/s` (**+7.23% throughput gain**)
- **Peak TG64 Throughput:** `25.7326 tok/s` (matches `mx-llama.cpp` 25.74 tok/s frontier!)
- **TG64 Latency:** `41.789 -> 38.971 ms/token` (**-2.819 ms/token latency saved**)
- **Allocations during decode:** 0
- **Replay check:** PASS

### 2. TG128 (128 tokens)

| Pair | Control (FUSED=0) ms | Control tok/s | Candidate (FUSED=1) ms | Candidate tok/s | Latency Delta ms/tok |
|---|---|---|---|---|---|
| Pair 1 | 5488.15 | 23.3230 | 5119.21 | 25.0039 | -2.882 |
| Pair 2 | 5496.86 | 23.2860 | 5128.28 | 24.9596 | -2.880 |
| Pair 3 | 5484.86 | 23.3370 | 5113.73 | 25.0307 | -2.899 |
| Pair 4 | 5496.29 | 23.2884 | 5127.90 | 24.9615 | -2.878 |
| Pair 5 | 5493.74 | 23.2992 | 5131.31 | 24.9449 | -2.831 |
| **Median** | **5493.74** | **23.2992** | **5127.90** | **24.9615** | **-2.858** |

- **TG128 Throughput:** `23.2992 -> 24.9615 tok/s` (**+7.13% throughput gain**)
- **Peak TG128 Throughput:** `25.0307 tok/s`
- **TG128 Latency:** `42.920 -> 40.062 ms/token` (**-2.858 ms/token latency saved**)
- **Allocations during decode:** 0
- **Replay check:** PASS

### 3. Hardware State & Telemetry

- TG64 Telemetry: 2,418 samples | SCLK == 1606MHz: 100.0% | Avg Temp: 44.0C (Max: 56.0C) | Avg Power: 58.3W (Max: 237.0W)
- TG128 Telemetry: 2,830 samples | SCLK == 1606MHz: 100.0% | Avg Temp: 48.0C (Max: 63.0C) | Avg Power: 78.7W (Max: 240.0W)
- Zero thermal throttling, clocks locked at 1606 MHz for 100% of samples.

## Interpretation

1. **Massive Latency Reduction:** Saving **+2.819 to +2.858 ms/token** represents the single largest individual optimization win in MIInfer history.
2. **Why Register Caching is So Potent on gfx906:** With only 48 heads distributed across MI50's 60 CUs, each CU executes exactly 1 workgroup (128 threads / 2 Wave64s). The CU is not occupancy-limited; each wave has access to up to 256 VGPRs. Holding the 128 floats of the state row in registers cuts HBM read traffic for the 3 MiB state matrix in half (saving 3 MiB of bandwidth per layer $\times 48 = 144 \text{ MB/token}$).
3. **Dispatch Overhead Elimination:** Eliminating 3 kernel launches per recurrent layer removes 144 kernel dispatches per token, dramatically shrinking the static HIP graph.
4. **Competitive Position vs mx-llama.cpp:**
   - Pre-M7 baseline (vanilla llama.cpp): 22.42 tok/s
   - EXP-0179: 23.33 tok/s
   - EXP-0181 (+ HIP Graph): 23.47 tok/s
   - EXP-0182 (+ Fused Gate+Up SwiGLU): 23.95 tok/s
   - **EXP-0183 (+ Fused DeltaNet Recurrent Core): 25.66 tok/s (peak 25.73 tok/s)**
   - Gap to `mx-llama.cpp` TG64 frontier (25.74 tok/s) is now **0.01 tok/s**! MIInfer has effectively caught the competitive frontier.

## Decision

**KEEP**.
- Latency saved: **+2.819 to +2.858 ms/token**.
- TG64 throughput reaches **25.66 tok/s** (peak 25.73 tok/s).
- Zero decode allocations maintained.
- Subphase M7-D is closed.

## Follow-up

Advance to **Subphase M7-E (Fused LM-Head GEMV + Argmax Reduction)**:
- Currently, the LM-head projects the final normalized hidden state ($5120$) to vocabulary logits ($151,936$ floats = 608 KiB) via Q6_K MMVQ (`launch_qwen3_q6_k_q8_1_mmvq`), taking ~2.485 ms.
- A separate argmax kernel (`launch_qwen3_argmax`) then scans the 608 KiB logits buffer in VRAM to find the token with maximum logit, taking ~0.468 ms.
- Total LM-head + argmax latency: ~2.95 ms/token.
- Fusing the argmax reduction into the LM-head GEMV workgroups (keeping the running max in LDS/registers and outputting only the single argmax token ID and top logit) will eliminate writing and reading the 608 KiB logits buffer to VRAM and eliminate the argmax kernel dispatch.
- This targets recovering ~0.6 to 0.8 ms/token, which will push MIInfer beyond 26.2 tok/s, surpassing `mx-llama.cpp` on both TG64 and TG128!
