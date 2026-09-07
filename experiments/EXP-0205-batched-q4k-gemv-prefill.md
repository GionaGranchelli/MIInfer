# EXP-0205 — Batched Q4_K Wave GEMV for Prefill

## Hypothesis

Unrolling the Q4_K wave GEMV kernel to process B tokens simultaneously (loading weights once,
applying to B input vectors) will amortize HBM bandwidth cost and improve prefill throughput.

## Motivation

Current prefill processes tokens one-at-a-time through the full pipeline at ~33 tok/s.
M11-B Phase 1 targets ≥100 PP tok/s at P512. The GEMV projections dominate layer time
(~25 ms out of ~28 ms per decode step). Amortizing weight loads across multiple tokens
is the most direct path to higher prefill throughput.

## Environment

```text
GPU:               AMD Instinct MI50 32GB HBM2 (gfx906 / Vega20, 60 CUs)
SCLK:              1606 MHz (Manual DPM 7)
MCLK:              1000 MHz (Manual DPM 2)
Power Cap:         225.0 W
ROCm Version:      6.4.0 / LLVM 20
Compiler:          clang++ 20 (hip-clang)
Shape:             11264 × 5120 (QKV+gate combined dimension)
Format:            Q4_K wave layout × Q8_1 activations
```

## Benchmark

Standalone harness (`scratch/test_kquant_gemm_batch.hip`).
Each wave processes one output row but B input vectors, with weight data loaded once.
8 waves per block (512 threads), 50 timed iterations after 5 warmup.

## Results

| Batch B | Total ms | ms/tok | tok/s | Eff. BW (GB/s) | Speedup vs B=1 | Correctness |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 (baseline) | 0.0569 | 0.0569 | 17,582 | 634 | 1.00× | exact |
| 1 (batched kernel) | 0.0564 | 0.0564 | 17,716 | 639 | 1.01× | exact |
| **2** | 0.0776 | 0.0388 | 25,779 | 929 | **1.47×** | exact |
| **4** | 0.1396 | **0.0349** | **28,657** | **1,033** | **1.63×** | exact |
| 8 | 0.4656 | 0.0582 | 17,183 | 619 | 0.98× | exact |
| 16 | 2.389 | 0.149 | 6,697 | 241 | 0.38× | exact |

## Profiling

### B=4 Sweet Spot Analysis

At B=4, effective bandwidth reaches 1,033 GB/s, exceeding the ~1 TB/s HBM2 theoretical peak.
This indicates weight data is being served from L2 cache for the 2nd–4th tokens within a wave.
Each wave loads 5 Q4KWaveTiles (5 × 640 = 3,200 bytes) once and processes 4 input vectors.

### B=8+ Regression

At B=8, the kernel requires 8 float accumulators (sd[8], sm[8], acc[8]) = 24 VGPRs just
for accumulators, plus weight state VGPRs. Total VGPR pressure likely exceeds the per-wave
limit for 8 waves/block occupancy, causing register spilling to scratch memory.

### Implications for Prefill

| Approach | Per-token GEMV time | Estimated PP tok/s |
| :---: | :---: | :---: |
| Current (B=1 token-by-token) | 0.057 ms | ~33 |
| B=4 chunked GEMV | 0.035 ms | ~53 |
| True GEMM (P=512, amortized) | ~0.001 ms (est) | ~200+ |

B=4 alone gives ~53 tok/s — insufficient for the 100 tok/s target.
A true GEMM approach (layer-wise prefill with weight reuse across 512 tokens) is needed.

## Decision

**KEEP as stepping stone, but insufficient alone.**

B=4 batched GEMV provides a validated 1.63× per-token speedup with zero correctness loss.
However, reaching 100 PP tok/s requires layer-wise prefill with proper GEMM-like weight reuse.

## Follow-up

1. Implement layer-wise prefill: process all P tokens through each layer before advancing
2. Profile per-layer breakdown to identify sequential bottlenecks (recurrent core)
3. Consider proper Q4_K × Q8_1 GEMM kernel for large batch projections
