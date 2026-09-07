# EXP-0214 — M11-B Native Q4_K Split-K Batched GEMM

## Hypothesis

A two-wave-per-output-row split-K kernel can process all B=128 input vectors
while loading each native Q4_K tile once per row, avoiding the full-matrix
FP16 conversion rejected by EXP-0213.

## Candidate

The temporary kernel assigned alternate Q4_K tiles to two waves per output
row, computed four prompt vectors at a time, wrote tile partials, and reduced
those partials in a second kernel. It was tested only on
`blk.8.ffn_down.weight` (5120 × 17408, Q4_K), with B=128.

## Environment

- GPU: gfx906 MI50/MI60-visible device
- ROCm: 6.4.0 / LLVM 20
- Model: `Qwen3.8-27B-Q4_K_M.gguf`
- Input: deterministic random Q8_1 vectors
- A/B: same process, same packed weights and inputs

## Correctness

The native split-K result matched the existing B=4 launch sequence with
`max_abs_error=9.54e-7` across all 128×5120 outputs.

## Results

| Candidate | Time | Throughput |
| --- | ---: | ---: |
| 32 existing B=4 launches | 14,287.8 ms | 8.96 tok/s |
| Native split-K two-wave GEMM | 8,712.85 ms | 14.69 tok/s |

The candidate is `1.64×` faster in this isolated projection test, but the
absolute result and the remaining non-projection work leave it well short of
the M11-B P512 target. No runtime integration or end-to-end claim was made.

## Decision

**REJECT for production integration.** The kernel is an exact, useful
mechanism result, but this decomposition is not sufficient to close the
40→100 tok/s end-to-end gap. The temporary kernel and benchmark were removed.

## Follow-up

Do not add another runtime path until a whole-pipeline projection plan is
measured; Q4_K split-K alone is not a sufficient M11-B gate strategy, and Q6_K
plus attention/recurrent preparation still require independent evidence.

## Re-evaluation — release build

The original isolated result was collected from a debug build and is not a
qualified performance claim. Repeating the same A/B harness with the
`mi50-release` target produced:

| Candidate | Time |
| --- | ---: |
| 32 existing B=4 launches | 9.48421 ms |
| Native split-K two-wave GEMM | 14.7235 ms |

The outputs remained exact (`max_abs_error=9.54e-7`), but the split-K mapping
was only `0.64×` the release B=4 control. The release result supersedes the
debug performance numbers; the rejection is strengthened.
