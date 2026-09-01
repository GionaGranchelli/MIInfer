# EXP-0018 — M5-C5b resident normalization weights

**Status:** KEEP  
**Milestone:** M5  
**Date:** 2026-09-01  
**Baseline commit:** `0e47c5b`  
**Candidate commit:** `10433be7f476`

## Hypothesis

Normalization tensors are immutable model data and are already present in the
GPU plan's persistent weight arena. Reading them there directly should remove
the redundant per-token host-to-device uploads without changing kernels,
launch geometry, or numerical behavior.

## Candidate

The Qwen3 GPU layer and final RMSNorm now validate and consume resident F32
normalization tensors through `Qwen3GpuPlan::device_tensor_data()`. The
temporary normalization-weight and head-weight buffers are no longer used by
the full decode workspace. Projection, normalization, attention, and cache
algorithms are otherwise unchanged.

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

The clean Release build passed all 19 CTest tests, including Q4/Q8 kernel
tests and GPU decode-sequence validation. The short and growing workloads
preserved the pinned finite greedy sequence. The position audit remained
deterministic with zero temporary allocations.

## Structural result

| Metric | C5a | C5b |
|---|---:|---:|
| Temporary allocations/token | 0 | **0** |
| Dispatches/token | 1,588 | **1,588** |
| Copy bytes/token | 3,315,200 | **2,082,304** |
| Synchronization sites/token | 795 | **650** |

## Performance result

| Workload | C5a | C5b | Change |
|---|---:|---:|---:|
| Short decode | 53.625 tok/s | **58.917 tok/s** | **+9.9%** |
| 64-token growing decode | 48.334 tok/s | **52.791 tok/s** | **+9.2%** |

Clean candidate artifacts:

```text
bench/results/20260901T083234Z-371025/  short
bench/results/20260901T083307Z-371777/  64-token growing decode
bench/results/20260901T083350Z-372617/  position audit
```

The C5a controls are retained under `bench/results/20260901T080644Z-365787/`
and `bench/results/20260901T080718Z-366543/`. Both measurements observed the
same approximately 930/350 MHz auto-mode state, so the comparison is valid
as a relative optimization result but is not a canonical 1725/1000 MHz
absolute benchmark.

## Interpretation

Resident normalization weights remove approximately 37.2% of the measured
per-token copy bytes and 145 synchronization call sites. The roughly 9%
end-to-end improvement confirms that this was a material fixed cost. The
unchanged dispatch count identifies dispatch/materialization as a separate
remaining optimization family.

## Decision

```text
KEEP — resident normalization weights
```

## Follow-up

1. Reprofile the C5b steady-state path without changing it further.
2. Re-establish validated clocks for canonical absolute measurements.
3. Select the next slice from the remaining dispatch, conversion, quantization,
   FFN, and LM-head costs.
