# EXP-0227 — M11-B Q4_K token-parallel row tile

## Hypothesis

One output row per block, with four Wave64s processing four tokens
concurrently from a single LDS-staged Q4_K row tile, will preserve token
parallelism without the large per-wave accumulator array rejected in prior
batch kernels.

## Motivation

EXP-0223 rejected split-K/two-wave row mapping, and EXP-0226 rejected a
four-row block that serialized a 32-token chunk. This candidate kept the
token work concurrent while staging only one row (`17 * 640 = 10.88 KiB`).

## Baseline

Eight production `launch_q4k_wave_gemv_batched4` calls for 32 tokens on the
exact Q4_K FFN Down tensor `[17408, 5120]`.

## Candidate

One 256-thread block per output row. Four waves each compute one token, then
the block advances to the next four-token group while reusing its LDS row
tile.

## Environment

- GPU: AMD Instinct MI50, gfx906
- Model: `Qwen3.8-27B-Q4_K_M.gguf`
- Tensor: `blk.8.ffn_down.weight`, Q4_K, `[17408, 5120]`
- Build: `mi50-release`
- Benchmark: `MIINFER_Q4K_TOKEN_TILE_BENCH=1 build/mi50-release/miinfer-q4k-layout-bench /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf`

## Correctness

32-token candidate output matched the eight-B4 output with max absolute error
`0`. All outputs were finite.

## Results

Interleaved baseline/candidate samples, five launches per timing sample:

| Path | Batch time |
| --- | ---: |
| Eight B4 launches | 2391.46 us |
| Token-tile candidate | 4907.97 us |

Candidate speedup: `0.48726x` (2.05x slower).

## Interpretation

The one-row-per-block mapping increases block count fourfold relative to the
production four-row mapping. Its LDS reuse is insufficient to pay for the
extra scheduling and staging work.

## Decision

REJECT.

Do not integrate this kernel or change the production prefill mapping.

## Follow-up

Further Q4_K prefill work needs a genuinely different skinny-GEMM execution
layout, not another row/tile remapping of the current Wave64 GEMV.
