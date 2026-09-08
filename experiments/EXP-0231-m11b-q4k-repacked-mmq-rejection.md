# EXP-0231 — M11-B Q4_K repacked MMQ tile

## Hypothesis

A three-plane Q4_K layout (nibbles, scale/minimum pairs, and superblock
constants) can feed a gfx906 16×16-thread MMQ tile and amortize FFN Down
weight traffic across prompt tokens.

## Motivation

The external gfx906 optimization uses this layout for a grouped prefill GEMM.
MIInfer tested the mechanism independently on the exact Qwen3.8-27B Q4_K FFN
Down tensor `[17408,5120]`, using the existing Q8_1 activation ABI and exact
Q4_K minimum correction.

## Baseline

Existing native Q4_K `launch_q4k_wave_gemv_batched4` launches, with the same
Q8_1 input blocks and interleaved timing.

## Candidate

One 256-thread kernel computes a 64-row × 64-token tile. Four weight
subblocks are staged per iteration in LDS; each thread accumulates four rows ×
four token outputs with gfx906 `sdot4`. The repacked weight footprint is
55,705,600 bytes for the tested tensor.

## Environment

- GPU: AMD Instinct MI50, gfx906
- Model: `Qwen3.8-27B-Q4_K_M.gguf`
- Tensor: `blk.8.ffn_down.weight`, Q4_K `[17408,5120]`
- Build: `mi50-release`
- ROCm: 6.4.0 / LLVM 20

## Correctness

The candidate matched the native batched kernel for both tested token counts:

| Tokens | Max absolute error |
| ---: | ---: |
| 4 | `4.77e-6` |
| 64 | `4.89e-6` |

The candidate uses an integer Q8 lane-sum reconstruction for the Q4_K minimum
term, matching MIInfer's exact production arithmetic rather than the rounded
Q8_1 `s` half.

## Results

| Tokens | Native B=4 control | Repacked MMQ | Candidate / control |
| ---: | ---: | ---: | ---: |
| 4 | `301.92 us` | `4174.59 us` | `0.0723×` |
| 64 | `4758.49 us` | `4228.89 us` | `1.125×` |

Measurements are medians of interleaved HIP-event samples. The B=4 control
contains one native batched launch; the B=64 control contains sixteen native
B=4 launches.

## Interpretation

The tile is useful only after the workload reaches its full 64-token tile. The
production layer-major executor intentionally uses B=4 because recurrent state
and causal continuation are advanced in that bounded chunk. Padding a B=4
chunk to 64 tokens wastes nearly all the MMQ work and regresses the exact
projection by about 13.8×.

The B=64 projection win is only 1.125×, far below the approximately 2.2×
whole-prefill improvement required to move the qualified P513 result from
about 45.9 tok/s to 100 tok/s. Exploiting it would require a new recurrent
chunk schedule, not a local projection change.

## Decision

**REJECT as a production change.** Keep the result as evidence for the
minimum useful batch size of a grouped Q4_K MMQ tile. Remove the prototype
from the production build.

## Follow-up

Only revisit grouped MMQ if a causally correct layer schedule can expose a
large dependency-free token tile and its whole-pipeline A/B result justifies
the additional workspace and numerical qualification.
