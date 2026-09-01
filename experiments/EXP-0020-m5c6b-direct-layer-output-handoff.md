# EXP-0020 — M5-C6b direct layer-output handoff

**Status:** KEEP (structural; performance neutral in measured A/B)  
**Milestone:** M5  
**Date:** 2026-09-01  
**Baseline:** `10433be7f476` (M5-C5b)  
**Candidate:** `d0d6311741aa`

## Hypothesis

The trace-free decode path needlessly copies each layer's final residual from
the persistent `layer_output` scratch buffer into the next ping-pong buffer.
Writing the residual directly to the caller-owned next buffer should remove 36
D2D copies and their associated synchronization sites without changing model
math, precision boundaries, or the diagnostic trace path.

## Candidate

When `capture_trace == false` and `output_device` is non-null, the final FFN
residual writes directly to `output_device`. The existing scratch-and-copy path
remains active for trace-producing and standalone execution. An explicit guard
rejects aliased input/output layer buffers.

The control is retained through `MIINFER_LAYER_OUTPUT_HANDOFF=copy`; the
production fast-path default is `direct`.

## Environment and workload

* Model: Qwen3-8B Q4_0, SHA256
  `458634762bea7dbe19f3ce0614465bafd15ee90e815229c427043afcf195d628`
* GPU: AMD Instinct MI50 / gfx906
* Build: Release, HIP 7.1.52802-9999
* Workload: prompt ID `14990`, 8 warmup tokens, 64 measured growing-context
  decode forwards, 3 balanced A/B pairs
* A/B order: `copy,direct,direct,copy,copy,direct`
* Hardware state: observed approximately 930 MHz SCLK / 350 MHz MCLK; rates
  are optimization evidence, not canonical-clock results.

## Structural result

The position audit at the candidate reports the predicted exact deltas:

| Metric | C5b control | C6b direct |
|---|---:|---:|
| Layer input/output D2D calls | 72 | **36** |
| Layer input/output D2D bytes | 1,179,648 | **589,824** |
| Total copy calls | 649 | **613** |
| Total copied bytes | 2,082,304 | **1,492,480** |
| Synchronization sites including finalization | 650 | **614** |
| Dispatches | 1,588 | 1,588 |
| Temporary allocations | 0 | 0 |

The remaining copies are 576 KV-cache writes and one final logits D2H copy,
plus 36 layer-input D2D copies.

## Correctness

Release CTest passed **19/19**. The interleaved control and candidate both
produced finite, deterministic, identical generated IDs, including the pinned
64-forward sequence.

## Performance

Raw result: `bench/results/20260901T085927Z-374528/`.

| Policy | Mean decode | Throughput |
|---|---:|---:|
| Copy control | 1257.443 ms / 64 | 50.896937 tok/s |
| Direct candidate | 1257.456 ms / 64 | 50.896418 tok/s |

The measured difference is effectively neutral (`0.99999x`) at the observed
low clocks. This is not reported as a throughput gain. The candidate is kept
because it removes a redundant operation, preserves the trace path, and makes
the steady-state ownership model more explicit at no measured cost.

## Decision

```text
KEEP — exact structural reduction; performance neutral in this A/B run
```

## Follow-up

The next isolated overhead candidates are direct/coarsened KV-cache writes and
GPU-side greedy argmax. Dispatch/materialization fusion remains separate.
