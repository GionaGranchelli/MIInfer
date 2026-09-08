# EXP-0228 — M11-B Q4_K four-token subwave mapping

## Hypothesis

Splitting each Wave64 into four 16-lane token subwaves will compute four
tokens concurrently while a four-row block reuses its LDS-staged Q4_K row
tiles. This is a small GEMM-like reduction change without a large batch
accumulator array.

## Motivation

EXP-0226 rejected serial token looping and EXP-0227 rejected one-row-per-block
token parallelism. This candidate changed the reduction width and retained
four rows per block.

## Baseline

Eight production `launch_q4k_wave_gemv_batched4` calls for 32 tokens on the
exact Q4_K FFN Down tensor `[17408, 5120]`.

## Candidate

Four rows per 256-thread block, four 16-lane subwaves per row, and one token
per subwave. Each subwave loops over all four Q4_K blocks in every tile and
reduces within width 16.

## Environment

- GPU: AMD Instinct MI50, gfx906
- Model: `Qwen3.8-27B-Q4_K_M.gguf`
- Tensor: `blk.8.ffn_down.weight`, Q4_K, `[17408, 5120]`
- Build: `mi50-release`
- Benchmark: `MIINFER_Q4K_SUBWAVE4_BENCH=1 build/mi50-release/miinfer-q4k-layout-bench /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf`

## Correctness

32-token candidate output matched within max absolute error `1.19209e-6`.
All outputs were finite.

## Results

Interleaved baseline/candidate samples, five launches per timing sample:

| Path | Batch time |
| --- | ---: |
| Eight B4 launches | 2383.52 us |
| Four-token subwave candidate | 9865.21 us |

Candidate speedup: `0.241608x` (4.14x slower).

## Interpretation

The 16-lane reduction requires each subwave to stride all four K blocks,
which greatly increases instruction work and loses the efficient Wave64
mapping. LDS row reuse does not recover that cost.

## Decision

REJECT.

Do not integrate this kernel or change the production prefill mapping.

## Follow-up

Close this Wave64 row/subwave remapping family. The remaining prefill path
needs a packed quantized GEMM with a different data layout and work
decomposition.
