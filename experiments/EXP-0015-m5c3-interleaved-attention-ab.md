# EXP-0015 — M5-C3 interleaved cached-attention A/B characterization

**Status:** KEEP (harness and relative result); RETEST (absolute rate at validated clocks)  
**Milestone:** M5  
**Date:** 2026-09-01  
**Baseline commit:** `c943ad49e21b`  
**Candidate commit:** `7b2ed8d6948d`  

## Question

Does the cooperative cached-attention path retain its measured advantage when
serial and parallel controls are run interleaved on the same loaded MI50
execution plan, without trace instrumentation?

## Hypothesis

Balanced in-process A/B ordering will reproduce the M5-C2 speedup while
reducing sensitivity to process startup, cache state, and machine drift.

## Workload and environment

```text
GPU: AMD Instinct MI50 / gfx906
Model: Qwen3-8B Q4_0
Model SHA256: 458634762bea7dbe19f3ce0614465bafd15ee90e815229c427043afcf195d628
Build: mi50-release
Prompt IDs: 14990
Warmup: 8 tokens (same trace-free semantics as M5-C0)
Measured decode: 64 growing-context forwards
Pairs: 3
Order: serial, parallel, parallel, serial, serial, parallel
```

The benchmark was run through `scripts/run-bench.sh`, which retained machine
state and active telemetry in:

```text
bench/results/20260901T061300Z-353809/
```

The result records commit `c943ad49e21b` and `git_dirty=false`. The telemetry
snapshots observed auto-mode clocks around 925–930 MHz SCLK and 350 MHz MCLK,
below the project's validated 1725/1000 MHz benchmark state. Therefore the
absolute tok/s values are hardware-state-qualified and are not a replacement
for the earlier high-clock baseline. Because both policies were interleaved
under the same run, their relative comparison remains the useful result here.

## Implementation

`miinfer-qwen3-attention-ab-bench` loads one model and plan, alternates the two
attention policies in balanced order, resets the KV cache before each sample,
and measures trace-free wall time. The serial policy is selected with
`MIINFER_ATTENTION_KERNEL=serial`; cooperative is selected with
`parallel`. The production kernel and numerical policy were not changed by
this experiment.

The cooperative position audit was rerun from the promoted M5-C2 build. Its
clean result is retained at:

```text
bench/results/20260901T-m5c3-position-audit/result.json
```

## Correctness

Both policies produced the same deterministic generated IDs, including the
pinned prefix:

```text
8, 341, 286, 470, 330, 9707, 11, 330
```

All measured outputs were finite and each run validated the expected final
KV-cache length.

## Results

| Policy | Mean decode | Throughput | Samples |
|---|---:|---:|---:|
| Serial control | 5011.723 ms | 12.770 tok/s | 3 |
| Cooperative | 1630.792 ms | 39.245 tok/s | 3 |

Relative result:

```text
3.073x cooperative-over-serial speedup
```

The cooperative position audit reported:

| Cache length | Production wall | Attention GPU time | Dispatches | Copy bytes |
|---:|---:|---:|---:|---:|
| 1  | 25.776 ms | 0.465 ms | 1,588 | 3,315,200 |
| 8  | 24.892 ms | 0.967 ms | 1,588 | 3,315,200 |
| 16 | 26.070 ms | 1.536 ms | 1,588 | 3,315,200 |
| 32 | 26.876 ms | 2.669 ms | 1,588 | 3,315,200 |
| 64 | 30.459 ms | 5.015 ms | 1,588 | 3,315,200 |

## Interpretation

The in-process interleaved result confirms that the cooperative kernel keeps
the M5-C2 advantage: it is about 3.1x faster than the serial control over the
64-forward growing-context workload. The position audit confirms that
attention remains context-dependent but no longer dominates the position-64
production wall time as it did under the serial kernel. Dispatch count and
copy bytes remain flat, so this experiment does not yet address launch or
materialization overhead.

## Decision

```text
KEEP — interleaved A/B harness and relative cooperative-attention result
RETEST — absolute throughput after restoring validated MI50 clocks
```

No production kernel change was made in M5-C3.

## Follow-up

1. Repeat the clean A/B run after a validated 1725 MHz SCLK / 1000 MHz MCLK
   state is available.
2. Use the cooperative path as the new baseline for the next optimization.
3. Characterize the remaining flat dispatch, quantization, FFN, and copy costs
   before selecting the next single-hypothesis optimization.
