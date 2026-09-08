# EXP-0245 — M11-B Q4_K MMQ16 split-4 rejection

## Hypothesis

A 16-token native Q4_K output-stationary kernel can avoid B=16 VGPR
accumulators by assigning four Wave64 lanes to each output cell. Each block
stages four output rows and 16 Q8_1 activation vectors in LDS, then reuses
each weight tile across the token group.

## Baseline and candidate

- Baseline: four existing native Q4_K B=4 launches for 16 tokens.
- Candidate: one 256-thread launch, four rows × 16 tokens, four cooperating
  threads per output cell, and staged native Q4_K weights/activations.

## Environment

- GPU: AMD Instinct MI50, gfx906
- Model: `Qwen3.8-27B-Q4_K_M.gguf`
- Tensor: exact Q4_K FFN Down `[17408, 5120]`
- Build: `mi50-release`
- Correctness: 20 warmup pairs and 51 interleaved HIP-event rounds

## Results

| Path | Median | Relative |
| --- | ---: | ---: |
| Four native B=4 launches | 1212.96 us | 1.000x |
| MMQ16 split-4 candidate | 2040.32 us | 0.594x |

The candidate matched the baseline within `1.90735e-6` maximum absolute error
and all outputs were finite.

## Interpretation

Reducing accumulator pressure did not recover the cost of the staged
16-token mapping and four-way partial reductions. This is another rejection
of the direct native 16-token remapping family, not evidence that a full
grouped GEMM is impossible.

## Decision

**REJECT.** The candidate, benchmark, declaration, and build target were
removed. Production execution is unchanged.

## Follow-up

Only pursue the remaining grouped-dataflow direction if it changes the
workgroup output tile and data layout materially; do not add another native
16-token Wave64 remapping.
