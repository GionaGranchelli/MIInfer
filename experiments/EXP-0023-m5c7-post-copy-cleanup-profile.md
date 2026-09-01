# EXP-0023 — M5-C7 post-copy-cleanup decode profile

**Status:** COMPLETE — recommendation: KERNEL  
**Milestone:** M5  
**Date:** 2026-09-01  
**Candidate under profile:** `ba5d0d3` / post-C6d path

## Goal

Measure the post-C6d steady-state decode path at P1 and P64, separating clean
production wall time from detailed operation-family timing and dispatch counts.
Use the result to choose between kernel optimization, fusion, and HIP graphs.

## Environment and workload

* Model: Qwen3-8B Q4_0, SHA256
  `458634762bea7dbe19f3ce0614465bafd15ee90e815229c427043afcf195d628`
* GPU: AMD Instinct MI50 / gfx906
* Build: Release, HIP 7.1.52802-9999
* Path: cooperative attention, persistent workspace, resident norm weights,
  direct layer handoff, coalesced KV store, GPU-side argmax
* Positions: 1 and 64, prompt token `14990`
* Hardware state: observed approximately 930 MHz SCLK / 350 MHz MCLK; rates
  are qualified structural/performance evidence.

## Results

Raw audit: `bench/results/20260901T094450Z-386101/`.

| Position | Clean wall | Whole-token GPU | Detailed audit wall | Detailed GPU events | Dispatches |
|---:|---:|---:|---:|---:|---:|
| 1 | 15.313 ms | 15.475 ms | 34.814 ms | 23.649 ms | 1,625 |
| 64 | 19.780 ms | 19.928 ms | 36.916 ms | 27.994 ms | 1,625 |

The lightweight whole-token HIP event tracks clean wall time closely: the
difference is 0.162 ms at P1 and 0.148 ms at P64. The detailed per-operation
event total is intentionally not compared directly with clean wall time; its
event recording and deferred-profile machinery perturbs the instrumented run.

### Operation-family profile

The detailed event timings and dispatch counts were:

| Family | P1 ms / dispatches | P64 ms / dispatches |
|---|---:|---:|
| Normalization | 2.880 / 289 | 3.038 / 289 |
| Quantization | 3.799 / 505 | 3.900 / 505 |
| Q/K/V projection | 1.651 / 108 | 1.710 / 108 |
| O projection | 0.818 / 36 | 0.830 / 36 |
| FFN projection | 6.892 / 108 | 7.205 / 108 |
| RoPE | 0.547 / 72 | 0.597 / 72 |
| Attention | 0.457 / 36 | 4.931 / 36 |
| Activation | 0.254 / 36 | 0.261 / 36 |
| Residual | 0.492 / 72 | 0.518 / 72 |
| Conversion | 2.240 / 324 | 2.287 / 324 |
| LM head | 2.864 / 1 | 2.876 / 1 |
| Argmax | 0.276 / 1 | 0.277 / 1 |
| KV store | 0.261 / 36 | 0.270 / 36 |

The topology is flat with context at 1,625 dispatches/token. Normalization,
quantization, and conversion account for 1,118 dispatches, or approximately
68.8% of the total. FFN projection is the largest individual compute family
at P64, but the lightweight whole-token timing shows that production wall is
already almost entirely device work rather than an unmeasured host-launch
gap.

## Correctness

The audit preserved the deterministic generated ID trajectory through the
audited range. Release CTest passed **19/19** after the profiling-only audit
change. C6d's full-logit/GPU-argmax A/B remains unchanged and green.

## Interpretation

HIP graphs are not the first target: there is no large production wall versus
device-timeline gap to recover, and graph capture would preserve the current
1,625-operation topology. The remaining work is dominated by actual GPU
execution, with FFN projections the largest individual family and
quantization/conversion/normalization the largest dispatch cluster.

## Decision

```text
KERNEL — begin with one measured FFN projection optimization. Revisit
quantization/conversion fusion only after the FFN A/B and a fresh profile.
```

No production change was made by this experiment.
