# EXP-0389 — P512 regression discriminator

## Question

Distinguish a current-HEAD P512 code regression from environment/runtime
instability before reopening B128 attribution. No optimization was authorized.

## Workloads and builds

GOOD was a clean Release build at `12fc127` in `/tmp/mi50-exp0389-good`.
CURRENT was the EXP-0389 start HEAD `1d9b6fe`. Both used HIP 7.1.52802-9999,
clang 20.0.0.rocm, the same model SHA
`7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`, hermetic
environment, `m25_hi_qualified`, fixed context 1024, and the exact repeated-fox
P512 prompt. mx was commit `2e9d29fe736969160f17476ec6f0a6298cee6966`.

Each binary used fresh processes in interleaved GOOD/CURRENT/mx order. No
profiler, graph, capture, scheduler experiment, B64, or B4 selector was set.

## Results

GOOD samples: `2481.97, 2814.95, 2915.14, 3210.51, 2496.07, 3480.56 ms`.
Median **2915.14 ms**; min/max **2481.97/3480.56 ms**; range **998.59 ms**.
Allocation: 2685 allocations, `21,993,243,028 B`; free VRAM
`11,484,004,352 B`.

CURRENT samples: `2451.00, 5055.65, 5158.60, 3568.50, 2539.51, 2989.85 ms`.
Median **3279.18 ms**; min/max **2451.00/5158.60 ms**; range **2707.60 ms**.
Allocation: 2685 allocations, `22,999,937,428 B`; free VRAM
`10,477,371,392 B`.

Pair ratios CURRENT/GOOD: `0.9875, 1.7960, 1.7696, 1.1115, 1.0174, 0.8596`.
The median delta is `364.04 ms` (+12.49%), but neither binary reproduces a
stable qualification sequence.

mx samples: `2308.749, 2314.646, 2308.652, 2307.467, 2306.844, 2309.011 ms`.
Median **2308.11 ms**; min/max **2306.84/2314.65 ms**; range **7.80 ms**.

## Contract comparison

The effective printed `m25_hi_qualified` selector vectors were identical.
The allocation delta reproduced exactly:

```text
22,999,937,428 - 21,993,243,028 = 1,006,694,400 B (~960 MiB)
```

This is a real resource-contract difference between revisions, but this
experiment does not identify its buffer or establish it as the cause.

Continuous telemetry recorded SCLK 1606 MHz and MCLK 1000 MHz throughout.
Junction temperature ranged 34–60 C. No throttle, ROCm error, stale MIInfer
process, or stale llama-bench process was observed. mx remained stable while
both MIInfer binaries varied.

## Decision

**Classification: MIINFER_RUNTIME_INSTABILITY.** GOOD also becomes slow and
variable, while mx remains stable. This rules out a clean source-only
current-HEAD regression as the present explanation. No bisect is authorized.

B64: **REJECTED — CURRENT M28 FRONTIER**.

B4: **BLOCKED**.

ONE next PRIMARY:

> Identify the minimal MIInfer runtime boundary responsible for run-to-run
> P512 variance, separating setup/model initialization, prefill GPU execution,
> and host synchronization while preserving the GOOD/CURRENT comparison.

Explicitly not authorized: performance optimization, B128 attribution, new
residual architecture, B64, B4, precision changes, padding, launch tuning,
MMQ tuning, buffer archaeology, or source bisect.

No optimization candidate was implemented. EXP-0389 only determined whether
the P512 baseline failure originates from source evolution or the common
MIInfer runtime/environment boundary.

## Provenance

Experiment commit and graph SHA are added after graph refresh. Final working
tree status is recorded in the provenance commit.
