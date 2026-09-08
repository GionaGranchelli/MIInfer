# EXP-0226 — M11-B Q4_K chunked row-tile projection

## Hypothesis

Keeping a complete Q4_K row tile resident in LDS while consuming a logical
batch of 32 tokens will amortize weight traffic and eight B4 dispatches enough
to improve prefill projection time.

## Motivation

EXP-0215 rejected a logical chunk that simply repeated the existing B4 kernel.
This candidate was the first true chunked dataflow: four output rows staged
once, then processed in four-token microbatches.

## Baseline

Eight `launch_q4k_wave_gemv_batched4` calls for 32 token vectors, using the
production Q4_K wave layout and the exact Q4_K FFN Down shape
`[17408, 5120]`.

## Candidate

One 256-thread launch. Each block stages four complete Q4_K rows into LDS
(`4 * 17 * 640 = 43.52 KiB`) and loops over the 32-token logical batch in
four-token microbatches.

## Environment

- GPU: AMD Instinct MI50, gfx906
- Model: `Qwen3.8-27B-Q4_K_M.gguf`
- Tensor: `blk.8.ffn_down.weight`, Q4_K, `[17408, 5120]`
- Build: `mi50-release`
- Benchmark: `MIINFER_Q4K_CHUNK_BENCH=1 build/mi50-release/miinfer-q4k-layout-bench /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf`

## Correctness

32-token candidate output matched the eight-B4 output with max absolute error
`0`. All outputs were finite.

## Results

Interleaved baseline/candidate samples, five launches per timing sample:

| Path | Batch time |
| --- | ---: |
| Eight B4 launches | 2386.14 us |
| Chunked LDS row tile | 7481.21 us |

Candidate speedup: `0.318951x` (3.14x slower).

## Profiling

No detailed counter capture was needed after the large regression. The
candidate's extra LDS staging and long per-row token loop did not compensate
for dispatch reduction.

## Interpretation

The useful reuse unit is not merely a persistent row tile. The current
Wave64-per-row arithmetic has insufficient batch parallelism to make this
serialized token loop competitive, and the 43.52 KiB LDS footprint adds
pressure without improving the measured projection.

## Decision

REJECT.

Do not integrate this kernel or change the production prefill chunk size.

## Follow-up

The next projection experiment must expose token parallelism inside the
microkernel, rather than adding a larger sequential token loop to the current
row mapping.
