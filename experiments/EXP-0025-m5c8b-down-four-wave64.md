# EXP-0025 — M5-C8b Down four-Wave64 GEMV candidate

**Status:** REJECT  
**Milestone:** M5  
**Date:** 2026-09-01  
**Baseline:** `b1d43d9fbf74`  
**Candidate:** `ee3de32a6f6f`

## Hypothesis

The Down projection has a long reduction dimension (`M=4096, K=12288`) and
measured approximately 13.7% lower logical bandwidth than Gate/Up. A
256-thread workgroup containing four independent Wave64 reductions, with one
output row per wave, might expose more memory-level parallelism while keeping
the existing per-row arithmetic and avoiding cross-wave reduction.

## Candidate

`zero-point-four-wave64` maps four output rows to one 256-thread workgroup.
Each Wave64 independently processes its row in 64-thread-strided Q4 blocks,
performs the existing Wave64 shuffle reduction, and writes its own output.
There is no LDS, cross-wave reduction, new quantization, fusion, or arithmetic
contract change. An exact-Q8 metadata counterpart is also retained for future
diagnostic use, but neither variant is production-selected.

## Environment and method

* GPU: AMD Instinct MI50 / gfx906 (`gfx906:sramecc+:xnack-`)
* Build: Release, HIP 7.1.52802-9999
* Hardware state: approximately 930 MHz SCLK / 350 MHz MCLK; rates are
  relative optimization evidence, not canonical-clock results
* Shapes: Gate/Up `12288×4096`, Down `4096×12288`
* Input: deterministic synthetic Q8_1 activation, seed `0x4D493050`
* Timing: 30 warmups, 300 HIP-event iterations, five control/candidate pairs
  per shape, interleaved control then candidate
* Raw JSON artifacts: `bench/results/20260901T104000Z-391000/`

## Results

| Projection | Control median | Candidate median | Control BW | Candidate BW | Candidate delta |
|---|---:|---:|---:|---:|---:|
| Gate | 50.240 µs | 51.200–51.280 µs | 564.1 GB/s | 552.7–553.5 GB/s | **+1.9% latency** |
| Up | 50.240 µs | 51.200 µs | 564.1 GB/s | 553.5 GB/s | **+1.9% latency** |
| Down | 57.279–57.280 µs | 83.360–84.800 µs | 494.7 GB/s | 334.1–339.9 GB/s | **+46.1% latency** |

All control and candidate runs passed the Q4_0 × Q8_1 quantized CPU oracle.
The candidate's five Down medians average approximately 83.71 µs, compared
with approximately 57.28 µs for the control. It therefore moves the target
farther from the C8b promotion threshold of 535 GB/s rather than toward it.

## Correctness

The candidate and its exact-metadata counterpart pass the expanded Q4/Q8
correctness matrix, including the small, tail, and all seven Qwen3-shaped
cases. The full Release suite remains green; no production kernel selection or
full-model precision behavior changed.

## Interpretation

More independent rows per workgroup did not improve the long-K path. The
candidate's four-wave workgroup reduces the number of workgroups but gives
each row only one Wave64 and six serial block iterations for Down. The current
256-thread per-row reduction remains substantially better for this shape. The
result rejects insufficient row-level workgroup parallelism as the next Down
optimization hypothesis; it does not establish the cause of the bandwidth gap.

## Decision

```text
REJECT — correctness passes, but Down latency regresses by about 46% and
Gate/Up also regress by about 1.9%. Do not promote the candidate.
```

## Follow-up

Stop this four-wave geometry family. C8c should investigate the Down-specific
long-K behavior using resource/instruction or split-K evidence, rather than
continue sweeping workgroup sizes. The production path remains unchanged.
