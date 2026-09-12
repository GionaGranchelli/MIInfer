# EXP-0324 — M25 Mx vectorized MMQ epilogue

**Status:** REJECT  
**Milestone:** M25  
**Date:** 2026-09-12  
**Baseline:** `a2f9cb4`  
**Candidate:** working tree only

## Hypothesis

Matching the pinned mx dense repacked-MMQ epilogue's `float4` stores would
reduce output-store overhead in MIInfer's default staged Mx kernel.

## Candidate

The legacy kernel's 16 scalar row stores per token were replaced by aligned
four-row `float4` stores, with scalar stores retained for edge rows. Weight
layout, arithmetic, tile geometry, and activation quantization were unchanged.

## Environment and benchmark

- AMD Instinct MI50 / gfx906 / Wave64
- Qwen3.8-27B-Q4_K_M, model SHA-256
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- Release `miinfer-m24-projection-bakeoff`, B512, resident Mx path
- `MIINFER_MX_PIPELINE` unset; SCLK/MCLK held at `1606/1000 MHz`
- each reported value is the harness median after warmup and repeated samples

## Correctness

The candidate remained finite and preserved the established CPU-contract
errors: Q4 `0.00000060`, Q5 `0.00000167`, and Q6 `0.00000048` maximum absolute
error.

## Results

| projection | scalar baseline (us) | vector-store candidate (us) | delta |
| --- | ---: | ---: | ---: |
| Q4 gate | 6829.592 | 7540.793 | +10.4% |
| Q5 SSM out | 2751.197 | 2981.277 | +8.4% |
| Q6 down | 8148.953 | 8917.112 | +9.4% |

The candidate regressed every tested production shape despite unchanged
numerical output.

## Decision

**REJECT.** Restore scalar stores and retain the current staged epilogue.
The pinned store form does not transfer to this MI50 kernel's register and
compiler balance; no end-to-end run was warranted.

## Follow-up

Do not revisit this without a new compiler/register-pressure measurement.
The remaining stretch work must target a genuinely different production
execution contract or a complete, correctness-preserving external path.
