# EXP-0016 — M5-C4 post-attention MI50 baseline

**Status:** RETEST  
**Milestone:** M5  
**Date:** 2026-09-01  
**Baseline commit:** `c943ad49e21b`  
**Candidate commit:** `7b2ed8d6948d` (cooperative attention)  

## Question

At the current cooperative-attention baseline, what are the absolute short and
growing-context decode rates, and does attention remain well behaved beyond
cache length 64?

## Scope

This is a measurement-only slice. It changes no production code. The short and
64-token workloads use the trace-free production path; the position audits add
deferred HIP events and are diagnostic rather than throughput measurements.

## Workload and model

```text
GPU: AMD Instinct MI50 / gfx906
Model: Qwen3-8B Q4_0
Model SHA256: 458634762bea7dbe19f3ce0614465bafd15ee90e815229c427043afcf195d628
Build: mi50-release
Prompt IDs: 14990
Short: warmup 1, measured 7 forwards, 3 runs
Growing: warmup 8, measured 64 forwards, 3 runs
Audit positions: 1,8,16,32,64 and 128,256,512,1024
```

## Results

Raw production results from clean commit `c943ad49e21b`:

```text
Short decode:   40.528 tok/s, 172.721 ms mean for 7 forwards
Growing decode: 38.094 tok/s, 1680.077 ms mean for 64 forwards
```

Artifacts:

```text
bench/results/20260901T062048Z-356125/
bench/results/20260901T062124Z-356897/
```

The short generated prefix and the 64-token sequence were deterministic and
finite. The cooperative position audit reported:

| Cache length | Production wall ms | Attention GPU ms | Quant ms | FFN ms | Dispatches | Copy bytes |
|---:|---:|---:|---:|---:|---:|---:|
| 1    | 24.363 | 0.467 | 4.080 | 6.563 | 1,588 | 3,315,200 |
| 8    | 25.251 | 0.966 | 4.097 | 6.805 | 1,588 | 3,315,200 |
| 16   | 26.178 | 1.537 | 4.081 | 6.741 | 1,588 | 3,315,200 |
| 32   | 27.169 | 2.671 | 4.215 | 6.845 | 1,588 | 3,315,200 |
| 64   | 29.486 | 5.027 | 4.145 | 6.862 | 1,588 | 3,315,200 |
| 128  | 30.528 | 9.481 | 3.966 | 6.767 | 1,588 | 3,315,200 |
| 256  | 39.662 | 18.529 | 3.944 | 6.774 | 1,588 | 3,315,200 |
| 512  | 57.977 | 36.844 | 4.021 | 6.835 | 1,588 | 3,315,200 |
| 1024 | 97.547 | 75.489 | 4.077 | 7.452 | 1,588 | 3,315,200 |

Audit artifacts:

```text
bench/results/20260901T0622-m5c4-position-audit/result.json
bench/results/20260901T0623-m5c4-long-audit/result.json
```

## Hardware validity

The benchmark runner captured the MI50 in auto mode with approximately
925–930 MHz SCLK and 350 MHz MCLK during the run. The required privileged
command to set 1725/1000 MHz was unavailable because sudo required a password.
Consequently these absolute rates are not accepted as the canonical
high-clock baseline. The interleaved relative result from EXP-0015 remains
valid because both attention policies ran under the same state.

## Interpretation

The cooperative attention curve remains smooth through cache length 1024; it
does not reproduce the serial kernel's pathological position-64 cost. At
cache length 1024, attention is 75.489 ms in the diagnostic pass while
quantization, FFN, dispatch count, and copied bytes remain approximately flat.
This confirms that attention is structurally fixed for the tested range, but
it remains the context-growing component. The new fixed-cost work is now more
visible in the production wall time and should be characterized before the
next optimization is selected.

## Decision

```text
RETEST — absolute production baseline after validated clock control
KEEP — cooperative attention as the production baseline
```

No optimization was accepted or rejected by this slice.

## Follow-up

1. Repeat the short and growing workloads at validated 1725/1000 MHz clocks.
2. Use the cooperative path as the M5-C5 baseline.
3. Select the next optimization from measured flat costs: dispatches,
   materialization/copies, quantization, and FFN projections.
