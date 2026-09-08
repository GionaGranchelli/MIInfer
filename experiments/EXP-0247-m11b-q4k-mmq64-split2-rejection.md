# EXP-0247 — M11-B Q4_K MMQ64 split-2 rejection

## Hypothesis

The rejected 4-thread output-cell MMQ64 mapping may be limited by per-cell
parallelism. Giving each output cell two cooperating threads and using a
512-thread block could reduce the inner reduction cost while retaining a
4-row × 64-token weight-reuse tile.

## Baseline and candidate

- Baseline: sixteen native Q4_K B=4 launches for 64 tokens.
- Candidate: one 512-thread launch per four output rows; two threads per
  output cell, Q4_K weights staged once per K tile, and four token groups.

## Environment

- GPU: AMD Instinct MI50, gfx906
- Model: `Qwen3.8-27B-Q4_K_M.gguf`
- Tensor: `blk.8.ffn_down.weight`, exact Q4_K shape `[17408, 5120]`
- Build: `mi50-release`
- Warmup: 20 baseline/candidate pairs
- Timing: 51 interleaved HIP-event samples per path

## Correctness

The candidate produced finite output and matched the native B=4 baseline with
maximum absolute error `3.57628e-6`.

## Results

| Path | Median | Relative |
| --- | ---: | ---: |
| Sixteen native B=4 launches | 4799.04 us | 1.000x |
| MMQ64 split-2 candidate | 41364.8 us | 0.116x |

## Interpretation

The larger 512-thread workgroup did not improve the grouped mapping; it was
8.6× slower than the existing B=4 sequence. The result is consistent with a
gfx906 occupancy/register or synchronization cliff, not a useful production
path.

## Decision

**REJECT.** The candidate, temporary benchmark, declaration, and build target
were removed. Production execution is unchanged.

## Follow-up

Stop enumerating simple native Wave64 output-cell remappings. The next useful
experiment must change the production schedule or use a genuinely different
GEMM decomposition.
