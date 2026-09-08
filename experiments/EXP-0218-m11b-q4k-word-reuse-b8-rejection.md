# EXP-0218 — M11-B Q4_K Word-Reuse B=8 Rejection

## Hypothesis

A specialized B=8 Q4_K kernel that decodes each weight word once, while
keeping eight output sums per wave, could improve the production-shaped
projection over two validated B=4 launches. This isolates a possible path
around EXP-0212's generic decoder-state B=8 regression.

## Baseline

Two `launch_q4k_wave_gemv_batched4` calls over eight Q8_1 activation vectors.
The comparison shape is the model's recurrent Q4_K FFN gate matrix:
5,120 input columns by 17,408 output rows.

## Candidate

One 256-thread launch using the existing four-row workgroup mapping and a
specialized word-reuse kernel. Each wave loads one output row's Q4_K tile and
accumulates eight batch results.

## Environment

- GPU: AMD Instinct MI60/MI50-visible gfx906
- SCLK/MCLK: 1606/1000 MHz
- ROCm: 6.4.0 / LLVM 20
- Build: `mi50-release`
- Model: `Qwen3.8-27B-Q4_K_M.gguf`
- Input: eight deterministic Q8_1 vectors
- Repetitions: 1 warmup, 8 batches, 2 iterations per batch

## Correctness

The candidate matched two B=4 launches with max absolute error
`1.43051e-6`.

## Results

| Mapping | Median | Relative |
| --- | ---: | ---: |
| Two B=4 launches | 482.608 us | 1.00x |
| One specialized B=8 launch | 761.183 us | 0.63x |

The generic B=8 mapping was also checked during the investigation on a
64-row slice and measured `0.57x` versus two B=4 launches; it does not rescue
the production mapping.

## Decision

**REJECT.** Correctness is exact, but eight live accumulators and the larger
instruction/register footprint outweigh the saved launch even when weight
decoding is reused. The runtime remains B=4.

## Follow-up

An accepted larger-batch projection needs a different output-stationary or
tile-staged decomposition, not another per-wave accumulator expansion.
