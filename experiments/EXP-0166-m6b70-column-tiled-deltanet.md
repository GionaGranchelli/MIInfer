# EXP-0166 — M6-B70 column-tiled DeltaNet state update

## Question

Can four Wave64s tile the transposed DeltaNet state by columns while retaining
the Qwen3.8 recurrence and improving memory access parallelism?

## Candidate

An opt-in gfx906 kernel used four waves per head block. Each wave updated one
state column and cooperated through shared partials for the per-row key dot and
query output. The existing transposed state layout and recurrence formula were
retained. Enable with `MIINFER_DELTA_COLUMN_TILES=1`.

## Result

The candidate built successfully but failed the native 16-token replay gate:

```text
M6-A23 failed: native generation replay mismatch
```

No performance benchmark was run. The candidate was removed before production
selection; the validated transposed no-decay-store LDS-input path remains
active.

## Decision

**REJECT.** The column-tiled recurrence changes the numerical execution
contract enough to fail autoregressive replay. Do not retry this mapping
without a separately validated state-layout/reduction contract.
