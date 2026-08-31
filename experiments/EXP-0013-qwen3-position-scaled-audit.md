# EXP-0013 — Qwen3 position-scaled execution audit

## Hypothesis

The large M5-C0 growing-context throughput loss is caused by work that scales
with KV-cache length, rather than by dispatch count, KV-history materialization,
or repeated activation quantization.

## Scope

This is an opt-in characterization only. It does not change production decode
semantics or kernel selection. The audit runs the same greedy trace-free decode
path at positions 1, 8, 16, 32, and 64. It performs a separate unprofiled pass
for production wall time, then a deferred-event pass for operation-family GPU
timing. Deferred events avoid the old M5-B per-operation synchronization, but
the audit pass still has event-recording overhead and is not a throughput
benchmark.

## Environment

```text
GPU: AMD Instinct MI50 / gfx906
Model: Qwen3-8B Q4_0
Model SHA256: 458634762bea7dbe19f3ce0614465bafd15ee90e815229c427043afcf195d628
Build: mi50-release
Prompt token: 14990
Positions: 1, 8, 16, 32, 64
```

Raw JSON is retained at:

```text
bench/results/20260831T204000Z-m5c1/result.json
```

## Implementation

`Qwen3GpuProfile` now supports deferred HIP-event resolution, copy-byte
accounting, synchronization counts, and a separate `kv_cache` copy category.
The new `miinfer-qwen3-position-audit` executable performs the selected
position audit while the existing trace-free benchmark remains unchanged.

## Results

The clean production-path wall time and non-intrusive category counters are:

| Position | Cache before | Production wall ms | Audit GPU ms | Attention GPU ms | KV-copy ms | Quant ms | FFN ms | Total copy bytes | Dispatches | Syncs | Temporary allocations |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1  | 1  | 26.471 | 27.861 | 3.401 | 3.974 | 3.809 | 6.741 | 3,315,200 | 1,588 | 795 | 1,086 |
| 8  | 8  | 35.050 | 34.757 | 12.439 | 3.929 | 3.807 | 6.742 | 3,315,200 | 1,588 | 795 | 1,086 |
| 16 | 16 | 44.828 | 45.268 | 22.819 | 3.946 | 3.834 | 6.821 | 3,315,200 | 1,588 | 795 | 1,086 |
| 32 | 32 | 66.394 | 67.301 | 44.132 | 4.284 | 3.994 | 6.986 | 3,315,200 | 1,588 | 795 | 1,086 |
| 64 | 64 | 119.253 | 118.879 | 95.998 | 4.081 | 3.955 | 6.795 | 3,315,200 | 1,588 | 795 | 1,086 |

The generated greedy trajectory was identical between the clean and audited
passes through the sampled range. The audit also reported finite logits at
every step.

## Interpretation

The dispatch count, total copied bytes, synchronization count, temporary
allocation count, quantization time, FFN time, and cache-write copy time are
flat across the sampled positions. Attention is not flat: it grows from
3.401 ms at cache length 1 to 95.998 ms at cache length 64. Production wall
time grows from 26.471 ms to 119.253 ms over the same points.

The causal statement is therefore:

> MInfer's context-scaling collapse is caused by the cached-attention kernel,
> not by growing dispatch count, rebuilding/copying the KV history, or
> context-dependent quantization. The current kernel launches one work item per
> query head and one thread performs the complete cache-length score and value
> scans, making the attention work explicitly O(context) with poor GPU
> parallelism.

This does not yet select a replacement kernel. It identifies the first
measured optimization target: improve cached-attention execution and memory
access at growing context, then benchmark the change against the clean M5-C0
control.

## Decision

`KEEP` as the M5-C1 characterization result. No production optimization is
accepted by this experiment.

## Follow-up

1. Establish fixed-context decode controls and an attention-only kernel
   baseline.
2. Prototype a better-parallel cached-attention implementation for gfx906,
   preserving the validated KV layout and numerical contract.
3. Run correctness and interleaved A/B throughput measurements against the
   trace-free M5-C0 benchmark.
4. Consider dispatch coarsening or HIP graph capture only after the
   context-scaling attention cost is addressed.
