# EXP-0249 — M11-B native Q4_K MMQ64 rows64 rejection

## Hypothesis

The only prior grouped projection that beat the B=4 control used a 64-row ×
64-token MMQ tile. Reusing that output tile with MIInfer's native Q4_K wave
layout should avoid the repack cost while retaining weight reuse. A 256-thread
block stages 64 Q4_K rows (about 40 KiB) and a 16-token activation slab, with
each thread computing a 2-row × 2-token sub-tile.

## Baseline and candidate

- Baseline: sixteen native Q4_K B=4 launches for 64 tokens.
- Candidate: one native-layout 256-thread launch per 64 output rows and 64
  tokens, with 64-row LDS weight staging.

## Environment

- GPU: AMD Instinct MI50, gfx906
- Model: `Qwen3.8-27B-Q4_K_M.gguf`
- Tensor: `blk.8.ffn_down.weight`, exact Q4_K shape `[17408, 5120]`
- Build: `mi50-release`
- Warmup: 20 baseline/candidate pairs
- Timing: 51 interleaved HIP-event samples per path

## Correctness

The candidate produced finite output and matched the native B=4 baseline with
maximum absolute error `1.0252e-5`.

## Results

| Path | Median | Relative |
| --- | ---: | ---: |
| Sixteen native B=4 launches | 4792.00 us | 1.000x |
| Native rows64 MMQ64 candidate | 9098.71 us | 0.527x |

## Interpretation

The native layout avoids repacking but its 16 live accumulators and repeated
native metadata/nibble work outweigh 64-row weight staging. The candidate is
slower than the B=4 control and cannot support production integration.

## Decision

**REJECT.** The candidate, temporary benchmark, declaration, and build target
were removed. Production execution is unchanged.

## Follow-up

The remaining grouped-MMQ evidence requires a layout and decomposition that
reduces native arithmetic/register pressure, not merely a larger native LDS
tile. Any future attempt must justify that new representation before coding.
