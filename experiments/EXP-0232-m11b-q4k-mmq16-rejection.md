# EXP-0232 — M11-B Q4_K 16-token MMQ tile

## Hypothesis

A smaller grouped Q4_K MMQ tile can expose enough token reuse for a causal
16-token chunk without requiring the 64-token tile rejected in EXP-0231.

## Motivation

EXP-0231 only became competitive at 64 tokens. This experiment tested the
middle ground against the exact Qwen3.8-27B FFN Down shape before changing the
production recurrent schedule.

## Baseline

Four interleaved native `launch_q4k_wave_gemv_batched4` launches for 16
token-major Q8_1 inputs.

## Candidate

A temporary 256-thread kernel staged 64 Q4_K rows per block in LDS and
computed 16 token outputs, using a three-plane repack with 160 bytes per
Q4_K block. The repack footprint was `55,705,600` bytes for
`[17408,5120]`.

## Environment

- GPU: AMD Instinct MI50, gfx906
- Model: `Qwen3.8-27B-Q4_K_M.gguf`
- Tensor: `blk.8.ffn_down.weight`, Q4_K `[17408,5120]`
- Build: `mi50-release`
- ROCm: 6.4.0 / LLVM 20
- Arithmetic: `MIINFER_KQUANT_FAST_ARITH=1`

## Correctness

The candidate matched the native control for 16 distinct activation rows;
maximum absolute error was `6.12e-6`.

## Results

Interleaved HIP-event medians over 101 rounds:

| Path | Median | Candidate / control |
| --- | ---: | ---: |
| Native Q4_K, four B=4 launches | `1191.78 us` | `1.000×` |
| Repacked MMQ16 | `3396.29 us` | `0.351×` |

The candidate was `2.85×` slower. Two isolated timing outliers were retained
in the raw result; they do not affect the medians.

## Interpretation

The 16-token tile does not amortize the LDS staging, scalar Q4 nibble
reconstruction, and per-token Q8 loads enough to beat the existing B=4
Wave64 mapping. EXP-0231's full 64-token result therefore cannot be reduced
to a production-sized causal tile by simple tiling.

## Decision

**REJECT.** The temporary repack and kernel were removed from the production
tree. Keep the result as negative evidence against an intermediate MMQ tile.

## Follow-up

Do not integrate a larger logical chunk around the current B=4 microkernel
without a new whole-pipeline causal schedule and an end-to-end A/B result.
