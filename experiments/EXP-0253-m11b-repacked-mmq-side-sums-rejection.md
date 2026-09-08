# EXP-0253 — Repacked MMQ64 with exact Q8 side sums

## Hypothesis

EXP-0252's exact int8 activation-sum reduction made the repacked MMQ64 kernel
near-parity at B64 but added substantial inner-loop work. Precomputing the
exact Q8 sums into a bounded side buffer should preserve correctness while
recovering the grouped-kernel throughput.

## Baseline

The current native `Q4KWaveTile` B4 kernel, launched once per four-token group,
on the exact Qwen3.8-27B Q4_K_M FFN-down shape `[17408,5120]`.

## Candidate

The EXP-0252 three-plane Q4_K repack and 64-row x 64-token MMQ tile, with an
additional int16 exact activation-sum buffer staged alongside Q8_1 inputs.
The candidate was standalone and not connected to the runtime.

## Environment

- AMD Instinct MI50, gfx906
- `Qwen3.8-27B-Q4_K_M.gguf`
- Q4_K FFN-down `[17408,5120]`
- Q8_1 activations
- `mi50-release`
- 51 interleaved A/B rounds per batch after warm-up

## Correctness

Maximum absolute error against the native B4 control was between `4.3e-6` and
`6.5e-6` for batches 4, 16, and 64.

## Results

| Batch | Native B4 (us) | Side-sum MMQ64 (us) | Candidate / baseline |
|---:|---:|---:|---:|
| 4 | 317.60 | 7342.88 | 0.043x |
| 16 | 1211.52 | 7322.08 | 0.165x |
| 64 | 4785.44 | 7374.56 | 0.649x |

## Interpretation

The side-sum path is substantially slower than the exact-reduction variant it
was intended to improve. The extra staged side-buffer traffic and occupancy/
register effects outweigh removal of the eight `sdot4` sum instructions. It
does not provide a production path toward the M11-B P512 gate.

## Decision

REJECT

## Follow-up

Remove the side-sum path. The remaining blocker is a causal grouped-dataflow
design that improves the deferred tail without another MMQ storage variant.
