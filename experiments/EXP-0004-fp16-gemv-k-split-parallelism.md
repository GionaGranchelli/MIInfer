# EXP-0004 — FP16 GEMV K-Split Parallelism

**Status:** PROPOSED  
**Milestone:** M1  
**Author:** MIInfer project  
**Date:** 2026-08-29  
**Baseline commit:** `b7dfbddce33f2dfd6d006bfe198a02a6f27dbf7f`  
**Candidate commit:** not implemented

## 1. Question

Does splitting each K/V output-row dot product across multiple workgroups
improve the complete `M=1024, K=4096` FP16 GEMV on the MI50 by exposing more
grid-level parallelism?

## 2. Hypothesis

The accepted baseline is grid-limited for K/V. Running two or four workgroups
per output row will improve total K/V latency despite the additional partial
sum reduction, because it increases the active grid from 1024 to 2048 or 4096
workgroups while preserving the real model dimensions.

## 3. Motivation

EXP-0003's synthetic M-sweep held `K=4096` fixed and measured approximately:

```text
M=1024  -> 1024 workgroups -> 119 GB/s
M=2048  -> 2048 workgroups -> 184 GB/s
M=4096  -> 4096 workgroups -> 381 GB/s
```

HOT versus STREAMING behavior was negligible, and the memory-streaming
reference did not explain the K/V result as an HBM ceiling. This motivates a
direct grid-parallelism test rather than reducing the number of workgroups by
placing multiple rows in each workgroup.

## 4. Source / Prior Art

This experiment is derived from the MIInfer EXP-0003 M-scaling diagnostic. It
does not import an external kernel or architecture-specific instruction.

## 5. Target Bottleneck

```text
Primary: grid-level parallelism / insufficient active work
Secondary: partial-result reduction and launch overhead (must be measured)
Evidence: EXP-0003 M-sweep and weak cache sensitivity
```

## 6. Baseline

The baseline is the accepted project-owned FP16 GEMV:

```text
Input: FP16
Weights: FP16 row-major M x K
Accumulator: FP32
Output: FP16
Workgroup: 256 threads, one workgroup per output row
Reduction: LDS partials and cooperative tree reduction
```

The accepted implementation and results are recorded in EXP-0002. The
baseline remains independently runnable and is configuration A below.

## 7. Candidate

The candidate changes only the amount of K-split work assigned to each output
row:

| Configuration | Workgroups/row | K slice/workgroup | Total workgroups |
| --- | ---: | ---: | ---: |
| A — baseline | 1 | 4096 | 1024 |
| B | 2 | 2048 | 2048 |
| C | 4 | 1024 | 4096 |

Each partial-dot workgroup writes one FP32 partial result. A final reduction
combines the partials for each row. The measured logical GEMV includes both
the partial-dot phase and that final reduction. The candidate must not add
runtime packing, DPP, swizzle, inline ISA, prefetch, or shape tables in this
experiment.

## 8. Hardware

```text
GPU: AMD Instinct MI50 32GB
Architecture: gfx906 / Vega20
Compute units: 60
Wave size: 64
Platform prerequisite: gfx802 isolated from KFD
```

Before each accepted run, verify `rocminfo` exposes gfx906 and no competing
MI50 workload is running. Record active SCLK, MCLK/HBM clock, temperature,
power, VRAM, and contamination status.

## 9. Model / Workload

```text
Operation: row-major FP16 GEMV
Input dtype: FP16
Weight dtype: FP16
Accumulator dtype: FP32
Output dtype: FP16
Cache regime: STREAMING, with HOT retained as a diagnostic if useful
```

The primary shape is the real Qwen3-8B K/V projection shape. Q is a real
Qwen3-8B guard/control shape.

## 10. Test Matrix

| ID | M | K | Origin | Purpose |
| --- | ---: | ---: | --- | --- |
| K-A | 1024 | 4096 | REAL MODEL SHAPE | baseline, 1 WG/row |
| K-B | 1024 | 4096 | REAL MODEL SHAPE | 2 WG/row |
| K-C | 1024 | 4096 | REAL MODEL SHAPE | 4 WG/row |
| Q-A | 4096 | 4096 | REAL MODEL SHAPE | baseline guard |
| Q-B | 4096 | 4096 | REAL MODEL SHAPE | 2 WG/row guard |
| Q-C | 4096 | 4096 | REAL MODEL SHAPE | 4 WG/row guard |

## 11. Correctness Method

Compare every configuration against the existing CPU FP32 accumulation oracle
using identical deterministic FP16 inputs and weights. Record max absolute
error, mean absolute error, meaningful max relative error, cosine similarity,
NaN/Inf status, and pass/fail. Existing EXP-0002 tolerances remain the
contract; performance from an incorrect candidate is invalid.

## 12. Benchmark

Allocate all buffers and create all HIP resources before timing. Time the
complete logical GEMV, including the partial-dot dispatches and final partial
reduction. Do not report the partial-dot kernel alone as the candidate result.

Use the accepted EXP-0002 streaming methodology, a pilot to choose stable
iteration counts, and at least five independent runs per configuration. Keep
raw per-sample results and environment/telemetry artifacts. Rotate order where
practical.

## 13. Acceptance

`KEEP` requires all of the following:

* all K and Q configurations are correctness-valid;
* the best K/V candidate improves complete-operation median latency by at least
  15% versus A;
* the improvement remains stable across five independent valid runs;
* persistent VRAM increase is negligible;
* clocks and other hardware state are valid;
* the result is reproducible from a clean candidate commit.

Q/O is a guard/control. It need not improve, but a material Q regression must
be recorded and prevents treating the candidate as a universal replacement.
If neither B nor C improves K/V after final reduction, the hypothesis is
`REJECT` and the next mechanism must target per-workgroup execution rather
than grid size.

## 14. Explicit Exclusions

This experiment does not implement or evaluate DPP reduction, DS swizzle,
inline gfx906 ISA, logical half-wave execution, vectorized packing, software
prefetch, autotuning, fusion, quantization, or runtime functionality.

## 15. Decision

Pending execution.

## 16. Follow-up

Select one subsequent experiment from the measured result. Do not implement a
follow-up optimization in EXP-0004.
