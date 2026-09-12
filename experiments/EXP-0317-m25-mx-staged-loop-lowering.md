# EXP-0317 — M25 staged MMQ loop lowering

**Status:** REJECT  
**Milestone:** M25  
**Date:** 2026-09-12  
**Baseline commit:** `e0bae93`  
**Candidate:** working tree only

## Question

Can small source-level ports of the pinned MMQ loop lowering close the
remaining P512 gap without changing the staged kernel contract?

## Candidates

Two isolated candidates were screened against the unchanged staged default:

1. Add `#pragma unroll` to the fixed group, row, and two-token loops in
   `mx_repacked_mmq_legacy_kernel`.
2. Hoist Q4/Q5 scale decoding out of the two-token loop and precompute the
   affine coefficients, with the analogous Q6 metadata hoist.

The loop shape and data layout otherwise remained unchanged. The source target
was the pinned `q8_repack/repack-kernels.cuh` implementation at
`2e9d29fe736969160f17476ec6f0a6298cee6966` (MIT; attribution in
`docs/references.md`).

## Environment and benchmark

AMD MI50/gfx906, Qwen3.8-27B-Q4_K_M, exact P512 H/I vector from EXP-0314,
`MIINFER_MX_Q8_BATCH=0`, `MIINFER_MX_PIPELINE` unset. Each candidate used the
same `18,472,649,044` device-byte allocation and completed finite P512 output.

## Results

| path | latency | throughput |
|---|---:|---:|
| unchanged staged default | 2435.75 ms | 210.20 tok/s |
| explicit unroll | 14575.97 ms | 35.13 tok/s |
| metadata hoist | 2534.59 ms | 202.01 tok/s |

The explicit unroll caused a catastrophic regression consistent with register
pressure/code-size expansion. Metadata hoisting was also slower than the
screening control. These runs were not full clock-qualified qualifications.

## Correctness

Both candidates completed finite P512 execution without a runtime error. No
candidate was promoted to the generation path.

## Decision

**REJECT both candidates.** Preserve the current implicit loop lowering. Any
future MMQ work needs gfx906 counter evidence first; copying CUDA unroll hints
or algebraic hoists is not sufficient.
