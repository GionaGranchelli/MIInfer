# EXP-0319 — M25-H/I configuration matrix

**Status:** KEEP; qualification gate passed  
**Milestone:** M25  
**Date:** 2026-09-12  
**Baseline commit:** `e0bae93`

## Question

Do the four advertised attention configurations work independently at the
targeted B128 and B512 widths?

## Matrix

| FFN | O | mode |
|---:|---:|---|
| 0 | 0 | control |
| 1 | 0 | `mx_ffn` |
| 0 | 1 | `mx_o` |
| 1 | 1 | `mx_ffn_o` |

## Environment and benchmark

AMD MI50/gfx906, Qwen3.8-27B-Q4_K_M, Release build, `MIINFER_MX_Q8_BATCH=0`,
`MIINFER_MX_PIPELINE` unset, and the hermetic environment established by
EXP-0318. Each mode ran in a fresh benchmark process with the same synthetic
layer input and scalar oracle. The benchmark reports GPU time for layer 3.

## Results — B128

| mode | GPU us | scalar max abs | scalar RMSE | live layer bytes |
|---|---:|---:|---:|---:|
| control | 19,748.461 | 0.012 | 0.001 | 375,369,616 |
| `mx_ffn` | 13,526.226 | 0.555 | 0.010 | 329,076,624 |
| `mx_o` | 18,932.783 | 0.479 | 0.015 | 395,653,008 |
| `mx_ffn_o` | 12,651.349 | 0.751 | 0.017 | 346,853,264 |

## Results — B512

| mode | GPU us | scalar max abs | scalar RMSE | live layer bytes |
|---|---:|---:|---:|---:|
| control | 63,391.624 | 0.033 | 0.001 | 561,950,608 |
| `mx_ffn` | 42,342.682 | 0.555 | 0.009 | 523,177,872 |
| `mx_o` | 60,833.229 | 0.566 | 0.015 | 589,754,256 |
| `mx_ffn_o` | 39,835.011 | 0.751 | 0.017 | 540,954,512 |

All outputs were finite and all four modes completed. The B512 matrix is a
functional/configuration gate, not an end-to-end throughput qualification.

## Decision

**KEEP** independent H/I operation. The O-only workspace bug is covered by the
`mx_o` rows. H/I remains opt-in until long-generation behavior and the
versioned preset are finalized.
