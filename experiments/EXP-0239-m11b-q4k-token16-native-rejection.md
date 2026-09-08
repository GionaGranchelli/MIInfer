# EXP-0239 — M11-B native Q4_K 16-token tile rejection

## Hypothesis

A native-layout Q4_K workgroup that reuses each weight tile across 16 Q8_1
activation vectors will expose useful token parallelism without dequantizing
weights or changing the causal runtime schedule.

## Baseline

The existing native Q4_K B=4 reuse kernel, invoked four times for 16 tokens,
is the production-shaped projection baseline. EXP-0238 established that the
current causal layer-major schedule can expose larger logical chunks, but its
projection launches remain B=4.

## Candidate

One 256-thread workgroup stages 16 rows of native Q4_K weights and one 16-token
Q8_1 input tile in LDS, then computes 16 output rows for all 16 tokens before
moving to the next 1024-weight tile. The candidate was benchmarked only as a
standalone exact-shape kernel; it was not integrated into production prefill.

## Environment

- GPU: AMD Instinct MI50, gfx906
- Model: `Qwen3.8-27B-Q4_K_M.gguf`
- Tensor: Q4_K FFN Down, `[17408, 5120]`
- Build: `mi50-release`
- ROCm: 6.4.0 / LLVM 20
- Tokens: 16
- Repetitions: 41 interleaved rounds, three launches per timing sample

## Correctness

The candidate output matched the existing B=4 path within
`max_abs_error=2.5034e-6` across all `16 * 5120` outputs. No non-finite
values were observed.

## Results

Median kernel time for the exact production-shaped tensor:

| Path | Time | Relative |
| --- | ---: | ---: |
| Four native B=4 launches | 1201.44 us | 1.000x |
| Native 16-token tile | 2619.89 us | 0.459x |

The candidate is 2.18x slower than the existing four-launch baseline. The
16-token accumulator footprint and LDS staging do not compensate for the
additional register pressure and reduced wave-level parallelism at this
skinny production shape.

## Decision

**REJECT.** Do not integrate this kernel or add a token-tile production flag.
The experiment closes the minimal native Q4_K 16-token reuse path; EXP-0238's
causal schedule remains unchanged.

## Follow-up

Any further M11-B projection work must demonstrate a materially different
dataflow or a measured end-to-end benefit before adding another kernel family.
