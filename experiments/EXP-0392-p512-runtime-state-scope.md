# EXP-0392 — P512 runtime-state scope

## Question

Determine whether the prior MIInfer P512 GPU variance is process-scoped,
run-scoped, first-use scoped, accumulating, or no longer reproducible.

## Why EXP-0391 is deferred

EXP-0391's recurrent/attention event records inflated host submission and
failed its perturbation gate. EXP-0392 therefore uses no profiler, HIP events,
family timing, or internal GPU instrumentation.

## Repeated-P512 harness contract

The diagnostic constructs one `Qwen35RuntimeEngine`, tokenizes the exact
repeated-fox P512 prompt once, then performs six normal `generate()` calls with
`max_new_tokens=0`, `stream=false`, and `reuse_session=false`. Each process is
fresh; iteration 1 is retained. The initial attempt exposed a preset-sanitizer
allow-list omission and was discarded. The corrected run preserved the repeat
selector and produced the matrix below.

## Process × iteration matrix (ms)

| Process | Run1 | Run2 | Run3 | Run4 | Run5 | Run6 | Median | Min | Max | Range |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2615.67 | 2524.84 | 2446.23 | 2467.67 | 2562.07 | 2552.20 | 2538.52 | 2446.23 | 2615.67 | 169.44 |
| 2 | 2558.89 | 2453.33 | 2976.41 | 2542.20 | 2494.88 | 2474.40 | 2518.54 | 2453.33 | 2976.41 | 523.08 |
| 3 | 2670.52 | 2521.08 | 2409.63 | 2545.81 | 2511.18 | 2545.90 | 2533.45 | 2409.63 | 2670.52 | 260.89 |
| 4 | 2823.53 | 2518.17 | 2498.68 | 2490.82 | 2521.53 | 2506.44 | 2512.31 | 2490.82 | 2823.53 | 332.71 |
| 5 | 2491.87 | 2599.92 | 2498.52 | 2517.89 | 2561.28 | 2584.13 | 2539.59 | 2491.87 | 2599.92 | 108.05 |
| 6 | 2777.44 | 2466.15 | 2496.20 | 2495.53 | 2426.95 | 2536.39 | 2495.87 | 2426.95 | 2777.44 | 350.49 |

Median within-process range: **296.80 ms**. Range of process medians:
**43.72 ms** (`2495.87–2539.59 ms`). Fastest process median: process 6,
`2495.87 ms`; slowest: process 5, `2539.59 ms`.

## Sequence-pattern classification

No process remained in the prior slow ~4.8–5.2 s state. No process flipped
between fast and slow states. There is no monotonic drift. Several processes
had a slower first run, but process 5 did not, and the first-use effect is not
consistent enough to explain the EXP-0389/0390 distribution. The one 2976.41
ms sample is an isolated excursion, not a stable process or run state.

## mx sentinel and telemetry

Four mx sentinels were `2312.21, 2313.31, 2315.39, 2314.48 ms`, range
`3.18 ms`; mx was stable. Corrected-run spot checks held SCLK 1606 MHz and
MCLK 1000 MHz with no observed throttle or stale process. Continuous telemetry
from the discarded overlapping attempt is not used as primary evidence; no
system-wide failure was observed during the valid matrix.

## Decision

**Classification: VARIANCE_NOT_REPRODUCED.** The 6×6 same-process matrix does
not reproduce the EXP-0389/0390 slow state. It cannot support PROCESS_SCOPED,
RUN_SCOPED, FIRST_USE, or ACCUMULATING_GPU_STATE claims. The aligned baseline
should be requalified before any deeper attribution.

B128 authorized: **NO**.

B64: **REJECTED — CURRENT M28 FRONTIER**.

B4: **BLOCKED**.

ONE next PRIMARY:

> Requalify the aligned current P512 baseline with a fresh clean matrix and
> only then decide whether P640/P896 B128 attribution is warranted. Do not
> resume family profiling unless the slow state reappears reproducibly.

Explicitly not authorized: optimization, kernel tuning, family profiling,
B128, residual architecture, B64, B4, or source bisect.

No optimization candidate was implemented. EXP-0392 only tested the lifetime
and scope of MIInfer's P512 GPU performance state.

## Provenance

Experiment commit and graph SHA are added after graph refresh. Final working
tree status is recorded in the provenance commit.
