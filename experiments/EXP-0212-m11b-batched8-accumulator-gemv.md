# EXP-0212 — M11-B B=8 Accumulator GEMV Rejection

## Hypothesis

Instantiating the existing four-row Wave64 GEMV with eight batch accumulators
could reuse the validated workgroup geometry and avoid the row-count penalty
of a one-row LDS mapping.

## Candidate

The existing `kquant_wave_gemv_batched_kernel` was instantiated with
`Batch=8` for Q4_K. No LDS redesign or runtime integration was used.

## Environment

- GPU: gfx906 MI50/MI60-visible device
- ROCm: 6.4.0 / LLVM 20
- Model shape: Q4_K, 5,120 rows × 17,408 columns
- Comparison: one B=8 launch versus two existing B=4 launches

## Correctness

The B=8 output was bit-identical to the two B=4 outputs (`max_abs=0`).

## Results

| Candidate | Time per B=8 workload | Relative |
| --- | ---: | ---: |
| Existing B=4 pair | 0.594705 ms | 1.00× |
| Generic B=8 accumulator | 0.861670 ms | 0.69× |

The additional accumulators/register state cost more than the saved launch.

## Decision

**REJECT.** Remove the temporary launcher and benchmark. The validated B=4
workgroup remains the production candidate.

## Follow-up

A viable next projection experiment must change the work decomposition enough
to improve memory reuse without simply multiplying per-wave accumulator state;
the next target is a measured skinny-GEMM mapping at the dominant model shapes.
