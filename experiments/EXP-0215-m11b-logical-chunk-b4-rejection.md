# EXP-0215 — M11-B Larger Logical Chunk with B=4 Inner Microtiles

## Hypothesis

Keep the validated B=4 kernels as inner microtiles but raise the bounded
layer-major logical chunk from 4 to 128 tokens. This should amortize layer
preparation, ping-pong traversal, and chunk-level dispatch work without
changing recurrent state ordering.

## Candidate

The temporary runtime accepted any multiple-of-four chunk up to 128, looped
the existing B=4 Q4_K/Q6_K launches across the chunk, and allocated reusable
per-layer buffers for 128 tokens. The default token-major path was unchanged.

## Environment

- GPU: AMD Instinct MI60/MI50-visible gfx906
- SCLK/MCLK: 1606/1000 MHz
- ROCm: 6.4.0 / LLVM 20
- Build: `mi50-release`
- Model: `Qwen3.8-27B-Q4_K_M.gguf`
- Graph capture: disabled for prefill comparisons

## Correctness

The P16 candidate produced the same 16-token `hello` continuation as the
token-major control. P4, P16, P129, and P513 prompt-ingestion checks completed
without a GPU fault. The multi-chunk checks were prompt-only; no long-
generation qualification was claimed.

## Results

| Workload | B=4 control | Logical-128 candidate |
| --- | ---: | ---: |
| P16 | 39.96 tok/s class | 40.68 tok/s |
| P128/P129 | 41.21 tok/s at P128 | 41.60 tok/s at P129 |
| P512/P513 | 39.96 tok/s at P512 | 40.26 tok/s at P513 |

The candidate is effectively neutral at production shape. It adds
approximately 2.6 GiB of persistent prefill workspace across the 48 recurrent
and 16 attention layers.

## Decision

**REJECT.** A larger logical chunk does not improve the existing B=4 schedule
enough to justify its memory cost. The temporary runtime change was removed.

## Follow-up

The remaining M11-B gap requires a materially different quantized projection
mapping or a proven whole-pipeline floor; chunk-size expansion alone is
exhausted.
