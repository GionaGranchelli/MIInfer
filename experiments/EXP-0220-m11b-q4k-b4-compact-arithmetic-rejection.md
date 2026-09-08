# EXP-0220 — M11-B Q4_K B=4 Compact Arithmetic Rejection

## Hypothesis

The validated Q4_K B=4 word-reuse kernel keeps temporary `sd[4]` and
`sm[4]` arrays in addition to four output sums. Accumulating dequantized
terms directly into the output sums might reduce register pressure and improve
the production layer-major path.

## Candidate

The Q4_K B=4 kernel was changed to accumulate each batch term directly into
`sums[4]`, removing the per-tile temporary arrays. No runtime schedule or
memory layout changed.

## Environment

- GPU: AMD Instinct MI60/MI50-visible gfx906
- SCLK/MCLK: 1606/1000 MHz
- ROCm: 6.4.0 / LLVM 20
- Build: `mi50-release`
- Model: `Qwen3.8-27B-Q4_K_M.gguf`
- Layer-major candidate with native Q4/Q5/Q6 projections and deferred tails

## Correctness

P128 and P513 runs completed with finite output and the existing candidate's
continuation behavior. Full CTest remained green after restoring the baseline
kernel.

## Results

| Workload | Candidate result |
| --- | ---: |
| P128, three runs | 45.91 / 45.83 / 45.76 tok/s |
| P513, three runs | 44.47 / 43.59 / 44.51 tok/s |

The P513 median was `44.47 tok/s`, below the committed B=4 kernel's
`45.02 tok/s` median.

## Decision

**REJECT.** Restore the temporary-array arithmetic. The measured B=4 kernel
mapping remains the best tested choice in this family.

## Follow-up

The remaining M11-B gap requires a different execution/dataflow strategy, not
another local accumulator rewrite.
