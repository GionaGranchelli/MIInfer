# EXP-0229 — M11-B Q4_K token-microtile wave decomposition

## Hypothesis

A logical 32-token projection can stage two output rows once and assign
separate waves to four-token microtiles. This should reuse the row tile while
avoiding the serial token loop in EXP-0226.

## Candidate

The candidate used a two-row LDS tile and four-token wave microtiles. Two
configurations were tested:

- 512 threads: two rows × four token groups per block
- 256 threads: two rows × two token groups per block, with more passes over
  the same 32-token chunk

Both kept four accumulators per wave and supported the exact Q4_K FFN Down
shape `[17408, 5120]`.

## Environment

- GPU: AMD Instinct MI50, gfx906
- Model: `Qwen3.8-27B-Q4_K_M.gguf`
- Tensor: `blk.8.ffn_down.weight`, Q4_K, `[17408, 5120]`
- Build: `mi50-release`
- Benchmark: `MIINFER_Q4K_TILE_MICROS_BENCH=1 build/mi50-release/miinfer-q4k-layout-bench /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf`

## Correctness

Both candidates matched the eight-B4 result with max absolute error `0`.
All outputs were finite.

## Results

Interleaved baseline/candidate samples, five launches per timing sample, batch
32:

| Mapping | Baseline | Candidate | Speedup |
| --- | ---: | ---: | ---: |
| 512 threads, 4 token groups | 2388.48 us | 3855.49 us | 0.619502x |
| 256 threads, 2 token groups | 2389.34 us | 3947.93 us | 0.605213x |

## Interpretation

Staging two rows in LDS and distributing B4 microtiles across waves does not
offset the extra LDS traffic and lower arithmetic efficiency. Reducing the
block to 256 threads does not help; the additional passes are more expensive
than the occupancy benefit.

## Decision

REJECT.

Do not integrate the tile-micro kernel or change the production prefill
mapping.

## Follow-up

The remaining quantized projection work needs a materially different packed
GEMM dataflow, likely with K-tile cooperation and explicit reuse of both
activation and weight fragments, rather than another B4 wave scheduling
variant.
