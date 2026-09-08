# EXP-0254 — Repacked MMQ64 exact-sum activation layout

## Hypothesis

The exact int8 activation-sum reduction in EXP-0252 and the side buffer in
EXP-0253 add overhead. Reusing the existing 36-byte Q8_1 footprint, with the
candidate-only half `s` field carrying the exact int16 sum, should preserve
strict correctness without extra LDS storage or sum instructions.

## Baseline

Native `Q4KWaveTile` B4 GEMV on the exact Q4_K FFN-down shape `[17408,5120]`.

## Candidate

The three-plane Q4_K repack and 64-row x 64-token MMQ tile from EXP-0252,
using a candidate-only exact-sum activation layout. The production Q8_1
layout was not changed and the candidate was standalone.

## Environment

- AMD Instinct MI50, gfx906
- `Qwen3.8-27B-Q4_K_M.gguf`
- Q4_K FFN-down `[17408,5120]`
- `mi50-release`
- 51 interleaved A/B rounds per batch after warm-up

## Correctness

Maximum absolute error against the native B4 control was `4.3e-6` to
`6.5e-6` for batches 4, 16, and 64.

## Results

| Batch | Native B4 (us) | Exact-layout MMQ64 (us) | Candidate / baseline |
|---:|---:|---:|---:|
| 4 | 318.24 | 5274.08 | 0.060x |
| 16 | 1214.88 | 5281.44 | 0.230x |
| 64 | 4785.76 | 5354.72 | 0.894x |

## Interpretation

Removing the exact-sum instruction and side-buffer costs improves the B64
candidate relative to EXP-0253, but it remains slower than the production B4
control. The grouped mapping does not produce a usable production projection
kernel on this MI50 shape.

## Decision

REJECT

## Follow-up

The repacked-MMQ and native token-reuse projection family is closed for M11-B.
The remaining work must change the causal layer dataflow or fuse a larger
post-state tail, not add another local Q4_K tile layout.
