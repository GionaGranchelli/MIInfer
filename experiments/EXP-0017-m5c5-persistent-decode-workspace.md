# EXP-0017 — M5-C5a persistent Qwen3 decode workspace

**Status:** KEEP  
**Milestone:** M5  
**Date:** 2026-09-01  
**Baseline commit:** `c943ad49e21b`  
**Candidate commit:** `2069783faf3d`

## Hypothesis

The steady-state decode path loses measurable time to allocating and freeing
temporary device buffers for every layer and token. Retaining one reusable
workspace for the decode session should remove that allocator churn without
changing kernels, launch geometry, or numerical precision boundaries.

## Candidate

`Qwen3GpuDecodeCache` now lazily owns a `Qwen3GpuDecodeWorkspace` shared by
the sequential layer loop and final LM-head path. It retains the existing
per-layer scratch buffers, conversion/Q8 buffers, attention score storage,
top-level ping-pong buffers, final-normalization buffers, logits, and Q6/K8
scratch. Standalone and teacher-forced layer diagnostics retain an owned local
workspace and do not share serving state.

Static norm-weight uploads remain unchanged and are intentionally deferred to
a separate experiment.

## Environment and workload

```text
GPU: AMD Instinct MI50 / gfx906
Model: Qwen3-8B Q4_0
Model SHA256: 458634762bea7dbe19f3ce0614465bafd15ee90e815229c427043afcf195d628
Build: mi50-release
Observed clocks: approximately 930 MHz SCLK / 350 MHz MCLK
Short: prompt token 14990, warmup 1, measured 7 forwards, 3 runs
Growing: prompt token 14990, warmup 0, measured 64 forwards, 3 runs
Attention: cooperative production default
```

## Correctness

Release CTest passed all 19 tests, including Q4/Q8 kernel tests and the
GPU decode-sequence test. The short and growing workloads produced the pinned
finite greedy sequence. The position audit remained deterministic and showed
the expected `1,588` dispatches and `3,315,200` copied bytes per token.

The audit's `temporary_allocations` counter changed from `1,086` per token in
the M5-C4 control to `0` at positions 1, 8, 16, 32, and 64.

## Results

Raw candidate artifacts:

```text
bench/results/20260901T080644Z-365787/       short
bench/results/20260901T080718Z-366543/       growing
bench/results/20260901T080500Z-364888/       position audit
```

| Workload | M5-C4 control | M5-C5a candidate | Change |
|---|---:|---:|---:|
| Short decode | 40.528 tok/s | **53.625 tok/s** | **+32.3%** |
| 64-token growing decode | 38.094 tok/s | **48.334 tok/s** | **+26.9%** |

The control artifacts are `bench/results/20260901T062048Z-356125/` and
`bench/results/20260901T062124Z-356897/`. Both control and candidate were
observed at approximately 930/350 MHz auto-mode clocks, so the comparison is
hardware-state qualified rather than a validated 1725/1000 MHz absolute
baseline.

## Position audit

The candidate's clean production wall times were:

| Cache length | Production wall ms | Attention GPU ms | GPU event ms | Temporary allocations |
|---:|---:|---:|---:|---:|
| 1  | 18.106 | 0.466 | 22.830 | 0 |
| 8  | 18.501 | 0.964 | 23.254 | 0 |
| 16 | 19.181 | 1.534 | 23.669 | 0 |
| 32 | 20.111 | 2.675 | 25.013 | 0 |
| 64 | 22.761 | 4.923 | 27.455 | 0 |

Dispatch count and copy bytes remained flat across the same positions. The
cooperative attention curve remains intact, so the workspace change did not
reintroduce the M5-C1 context-scaling defect.

## Interpretation

The zero-allocation invariant and the end-to-end improvement support the
hypothesis. Because the candidate changes only buffer lifetime and ownership,
the result is attributable to removing repeated device allocation/free work,
subject to the hardware-state qualification above.

This slice does not claim that all fixed overhead is removed: static norm
weights are still uploaded during execution, and dispatch count/materialized
copy bytes are unchanged.

## Decision

```text
KEEP — persistent decode workspace
```

## Follow-up

1. Re-establish the validated high-clock state for a canonical absolute
   throughput retest.
2. Profile the new steady-state path.
3. Run a separate A/B for persistent GPU-resident norm weights.
4. Revisit dispatch/materialization fusion only after the new baseline is
   recorded.
