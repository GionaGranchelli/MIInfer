# EXP-0014 — Cooperative cached-attention execution

## Hypothesis

The M5-C1 context-scaling collapse is caused by the current cached-attention
kernel assigning one active thread to each query head. A 256-thread workgroup
per query head should expose parallelism across the head dimension and across
cache positions without changing the persistent KV-cache layout.

## Baseline

The serial control is `qwen3_cached_attention_kernel`: one workgroup per query
head, one active thread, serial score/value scans over the full cache. M5-C0
trace-free results for the pinned Qwen3-8B Q4_0 workload were:

```text
short:   31.006 tok/s  (warmup 1, measured 7, 3 runs)
growing: 14.430 tok/s  (warmup 0, measured 64, 3 runs)
```

## Candidate

`qwen3_cached_attention_parallel_kernel` uses one 256-thread workgroup per
query head. Threads cooperatively reduce each query/key dot product, compute
the softmax denominator, and independently accumulate output dimensions. It
reads the existing `[kv_head][position][head_dim]` cache layout and preserves
the existing F32 attention output followed by the validated FP16 materialization
boundary. The candidate is selected by default; `MIINFER_ATTENTION_KERNEL=serial`
retains the baseline control.

## Environment

```text
GPU: AMD Instinct MI50 / gfx906
Model: Qwen3-8B Q4_0
Model SHA256: 458634762bea7dbe19f3ce0614465bafd15ee90e815229c427043afcf195d628
Build: mi50-release
Prompt: token 14990
```

## Correctness

The existing GPU multi-token sequence test passed with the candidate:

```text
14990 → 8 → 341 → 286 → 470 → 330 → 9707 → 11 → 330
```

Replay determinism passed. The candidate position audit reported finite logits
and the same 65 selected next-token IDs through position 64 as its unprofiled
control pass. The existing Release CTest suite remained green before and after
promotion.

## Results

### Position-scaled audit

| Position | Serial attention ms | Cooperative attention ms | Serial production wall ms | Cooperative production wall ms |
|---:|---:|---:|---:|---:|
| 1  | 3.405 | 0.455 | 26.603 | 24.893 |
| 8  | 12.446 | 0.957 | 35.387 | 25.081 |
| 16 | 22.827 | 1.525 | 45.730 | 25.627 |
| 32 | 43.965 | 2.664 | 66.554 | 27.109 |
| 64 | 95.976 | 4.948 | 119.307 | 27.802 |

The audit uses deferred HIP events and is not itself a throughput benchmark;
production wall values come from a separate trace-free pass.

### End-to-end trace-free A/B

| Workload | Serial control | Cooperative candidate | Change |
|---|---:|---:|---:|
| Short, warmup 1 + measured 7 | 31.006 tok/s | 40.267 tok/s | +29.8% |
| Growing, warmup 0 + measured 64 | 14.430 tok/s | 38.754 tok/s | 2.69× |

The candidate reduced the measured context-64 production wall from the serial
audit control's approximately 119 ms at the selected point to approximately
28 ms. It also preserved the exact pinned short generated prefix.

## Interpretation

The result confirms that cached attention was the context-dependent bottleneck.
At cache length 64, cooperative attention is approximately 19× faster in the
position audit while dispatch count and copy traffic remain unchanged. The
candidate removes the one-active-thread bottleneck and parallelizes the
head-dimension reductions and value accumulation.

This is a kernel/execution-geometry result, not a claim that attention is now
fully optimized. The candidate performs repeated workgroup barriers for each
cache position and remains a deliberately simple baseline for future layout,
fusion, or graph experiments.

## Decision

```text
EXP-0014 KEEP
M5-C2 KEEP
```

Promote the cooperative kernel as the production default. Retain the serial
kernel behind `MIINFER_ATTENTION_KERNEL=serial` as a reproducible control.

## Follow-up

1. Re-run the full benchmark suite with repeated interleaved serial/candidate
   measurements and hardware telemetry.
2. Profile the cooperative kernel's barrier and cache-read behavior at longer
   contexts.
3. Consider cache-layout/coalescing and graph-capture experiments only after
   this correctness-preserving speedup is established as the new baseline.
