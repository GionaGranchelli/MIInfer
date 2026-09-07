# EXP-0195: 1-Wave-Per-Row 0-LDS Fused SwiGLU Kernel Evaluation

## Hypothesis
In `launch_q4k_wave_fused_gate_up_swiglu`, switching from a 2-wave-per-row cooperative workgroup (with LDS reduction and `__syncthreads`) to a 1-wave-per-row execution model (4 rows per 256-thread block, 0 bytes LDS, intra-wave `__shfl_down` only) will eliminate wave load imbalance across 5 tiles, eliminate LDS allocations, and reduce decode latency across all 64 layers.

## Motivation
In the M8 baseline, `q4k_wave_fused_gate_up_swiglu_kernel` assigns 2 waves (128 threads) to each row, where Wave 0 computes tiles 0, 2, 4 (3 tiles) and Wave 1 computes tiles 1, 3 (2 tiles), followed by an LDS exchange via `odd_gate[2]` and `odd_up[2]`. While a microbenchmark on isolated buffers showed a slight reduction in kernel time (93.78 us -> 93.20 us), the behavior in the full end-to-end HIP graph execution across 64 layers needed empirical validation.

## Baseline (Control)
- Commit: M9 post-EXP-0194 (`MIINFER_SWIGLU_1WAVE=0`)
- Model: `Qwen3.8-27B-Q4_K_M.gguf`
- Hardware: AMD Instinct MI50 32GB (gfx906 / Vega20, 60 CUs, locked 1606 MHz SCLK / 1000 MHz MCLK, 225W cap)
- Execution: 64-layer GPU hybrid pipeline, HIP Graph mode

## Candidate
- `q4k_wave_fused_gate_up_swiglu_1wave_kernel<5>` (`MIINFER_SWIGLU_1WAVE=1`)
- Grid: `(rows + 3) / 4`, Block: 256 threads (4 waves)
- LDS: 0 bytes
- Reduction: Intra-wave shuffle only

## Verification
- CTest: 21/21 PASS
- Observable contract: 64/64 teacher-forced token match PASS, `logits_cosine = 0.999615` (>= 0.9995 requirement), argmax rank 1.
- Exact bitwise match in microbenchmark (`max_diff = 0`).

## Benchmark Results (TG64)
- **Control (`MIINFER_SWIGLU_1WAVE=0`):**
  - Warmup: 2133.89 ms
  - Samples: 2115.94, 2120.50, 2120.97, 2125.66, 2126.91 ms
  - **Median:** 2120.97 ms (30.175 tok/s, 33.140 ms/tok)
  - Allocations during decode: 0
  - Replay: PASS

- **Candidate (`MIINFER_SWIGLU_1WAVE=1`):**
  - Warmup: 2187.85 ms
  - Samples: 2176.42, 2181.16, 2181.22, 2183.95, 2187.59 ms
  - **Median:** 2181.22 ms (29.341 tok/s, 34.082 ms/tok)
  - Allocations during decode: 0
  - Replay: PASS

- **Delta:** -0.834 tok/s (-2.76% throughput regression, +0.942 ms/token latency penalty).

## Analysis & Root Cause
On gfx906 / Vega20, the 2-wave cooperative structure allows Wave 0 and Wave 1 within each workgroup to overlap their respective HBM load latencies for alternating tiles (e.g. Wave 0 fetching Tile 0 while Wave 1 fetches Tile 1). In the 1-wave model, each single wave must issue all 5 tile loads sequentially; without inter-wave latency hiding within the workgroup, memory stalls increase during full-graph replay. Furthermore, the 2-wave model has twice the thread density per row to consume the incoming memory stream.

## Decision
**REJECT for default execution.**
Environment toggle `MIINFER_SWIGLU_1WAVE` defaulted to disabled (`0`).

## Follow-up
Pursue Opportunity 1 (Combined Recurrent QKV + Gate Projection) and Opportunity 2 (Combined Attention Q + K Projection), which preserve the proven 2-wave GEMV cooperative tile geometry while amortizing kernel launch overhead and improving wave occupancy.
