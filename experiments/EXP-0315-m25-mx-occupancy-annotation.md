# EXP-0315 — M25 MMQ occupancy annotation

**Status:** REJECT  
**Milestone:** M25  
**Date:** 2026-09-12  
**Baseline commit:** `e0bae93`  
**Candidate:** working tree only

## Question

Does matching the pinned mx MMQ launch annotation of two resident blocks per
CU improve MIInfer's staged Mx Q4/Q5/Q6 kernel?

## Hypothesis

The pinned kernel uses `__launch_bounds__(256, 2)`, while MIInfer's staged
kernel used `__launch_bounds__(256, 1)`. Matching it might expose another
resident workgroup on gfx906.

## Candidate

Change `mx_repacked_mmq_legacy_kernel` from `__launch_bounds__(256, 1)` to
`__launch_bounds__(256, 2)`. No other source or runtime setting changed.

## Environment and benchmark

AMD MI50/gfx906, Qwen3.8-27B-Q4_K_M, exact P512 prompt and qualified H/I
vector from EXP-0314, `MIINFER_MX_Q8_BATCH=0`. Both runs allocated
`18,472,649,044` device bytes. The paired runs observed SCLK/MCLK at
`1606/1000 MHz`.

## Results

| path | latency | throughput |
|---|---:|---:|
| staged default, one block annotation | 2435.75 ms | 210.20 tok/s |
| two-block annotation | 2679.44 ms | 191.08 tok/s |

The candidate was 9.1% slower in this paired screening run.

## Correctness

The candidate completed finite P512 output without a runtime error. No
generation correctness claim is made from this performance screen.

## Decision

**REJECT.** Keep the one-block annotation. A revisit requires VGPR, LDS,
occupancy, and instruction-counter evidence; changing the annotation alone is
not useful on this gfx906 lowering.
