# EXP-0021 — M5-C6c coalesced KV-cache writes

**Status:** KEEP  
**Milestone:** M5  
**Date:** 2026-09-01  
**Baseline:** `d0d6311741aa` (M5-C6b)  
**Candidate:** `a202a946a8ac`

## Hypothesis

Each decode token currently performs 576 synchronous 512-byte D2D copies to
append K and V heads to the persistent cache. One device-side store launch per
layer can preserve the existing `[kv_head][position][head_dim]` FP32 layout
while removing the copy API/synchronization explosion.

## Candidate

The production default now launches one 256-thread KV-store kernel per layer.
Each thread writes one K and one V element to the existing cache destinations.
The per-head `hipMemcpy` implementation remains available as the control via
`MIINFER_KV_CACHE_WRITE=copy`; `store` is the default.

## Environment and workload

* Model: Qwen3-8B Q4_0, SHA256
  `458634762bea7dbe19f3ce0614465bafd15ee90e815229c427043afcf195d628`
* GPU: AMD Instinct MI50 / gfx906
* Build: Release, HIP 7.1.52802-9999
* Workload: prompt ID `14990`, 8 warmup tokens, 64 measured growing-context
  decode forwards, 3 balanced A/B pairs
* A/B order: `copy,store,store,copy,copy,store`
* Hardware state: observed approximately 930 MHz SCLK / 350 MHz MCLK; rates
  are optimization evidence, not canonical-clock results.

## Structural result

The candidate position audit reports:

| Metric | C6b / copy control | C6c / store |
|---|---:|---:|
| KV memcpy calls | 576 | **0** |
| KV memcpy bytes | 294,912 | **0** |
| Total copied bytes | 1,492,480 | **1,197,568** |
| Synchronization sites | 614 | **38** |
| Dispatches | 1,588 | **1,624** |
| Temporary allocations | 0 | 0 |

The additional 36 dispatches are the one store launch per layer. The remaining
copy traffic is 36 layer-input D2D copies and one final logits D2H copy. Cache
layout, append ordering, reset behavior, and snapshots remain unchanged.

## Correctness

Release CTest passed **19/19**. The interleaved control and candidate both
produced finite, deterministic, identical generated IDs through the complete
64-forward workload.

## Performance

Raw clean result: `bench/results/20260901T091213Z-377343/`.

| Policy | Mean decode | Throughput |
|---|---:|---:|
| Copy control | 1236.517 ms / 64 | 51.758283 tok/s |
| Coalesced store | 1172.151 ms / 64 | 54.600468 tok/s |

The candidate is `1.05491x` / approximately **5.5% faster** in the balanced
interleaved run. Clocks were in the observed low auto-mode state, so this is a
relative optimization result rather than the canonical absolute MI50 rate.

## Decision

```text
KEEP — exact KV representation preserved; 576 tiny copies collapsed to 36
device launches; approximately 5.5% end-to-end gain
```

## Follow-up

GPU-side greedy argmax is the next isolated copy reduction. Dispatch and
materialization fusion should be evaluated after that fresh profile.
