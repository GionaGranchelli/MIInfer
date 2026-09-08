# EXP-0241 — M11-B Q4_K row-major 16-token tile rejection

## Hypothesis

The EXP-0239/0240 mappings kept multiple token accumulators live in each
thread or wave. Assigning one 1024-thread workgroup to one output row and all
16 tokens should remove that pressure while reusing the row's native Q4_K
weight tile across the token batch.

## Baseline

Four native Q4_K B=4 launches for 16 tokens, the production-shaped baseline.

## Candidate

One 1024-thread block stages one Q4_K row tile and 16 Q8_1 activation vectors
per K tile. Each wave owns one token and accumulates one output row. The
candidate was standalone only and was not integrated into causal prefill.

## Environment

- GPU: AMD Instinct MI50, gfx906
- Model: `Qwen3.8-27B-Q4_K_M.gguf`
- Tensor: Q4_K FFN Down, `[17408, 5120]`
- Build: `mi50-release`
- ROCm: 6.4.0 / LLVM 20
- Tokens: 16
- Repetitions: 41 interleaved rounds, three launches per timing sample

## Correctness

The candidate matched the B=4 path within `max_abs_error=2.5034e-6` across all
outputs. No non-finite values were observed.

## Results

| Path | Time | Relative |
| --- | ---: | ---: |
| Four native B=4 launches | 1198.67 us | 1.000x |
| Row-major 16-token tile | 2027.04 us | 0.591x |

The candidate is 1.69x slower. Removing token accumulators does not offset the
1024-thread block and per-row workgroup schedule at this skinny shape.

## Decision

**REJECT.** Remove the standalone kernel and benchmark; do not integrate a
row-major token-tile flag.

## Follow-up

EXP-0239, EXP-0240, and EXP-0241 close three direct native-layout 16-token
Q4_K mappings. Further work must change the quantized dataflow or fuse a
larger portion of the recurrent tail rather than reshuffle the same GEMV.
