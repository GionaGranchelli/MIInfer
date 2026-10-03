# M29-0006 physical KV view

Status: `M29_5_N1_PHYSICAL_VIEW_INTEGRATED_GENERATION_GATE_BLOCKED`

Contract base: `78adc52c85c05095e6cd0e5a034690b4df2b3b91`

Contract commit: `b71d08600c586927a7b4c07bbb3c7eef15a57d44`

N=1 integration commit: `f6284ec`

## N=1

`PhysicalKvView` is now the explicit kernel-facing type at all four attention
entry points (forward, decode, and their profiled variants). Production
`AttentionLayerKvCacheStorage` allocates one placement-owned `DeviceKvPool`
block, registers a full-capacity `ContextSpace` page and `PlacementPlan` shard
for all KV heads, and caches the resulting raw-pointer view. No per-element
logical lookup, allocation, copy, or synchronization was added.

The Z840 alternate endpoint `100.118.66.80` is reachable. The physical-view,
HIP smoke, ContextSpace, and placement tests pass there. The N=1 exit remains
blocked by the model-level generation gate below.

The production PrefillV2 end-to-end harness was also run on the Z840 candidate
and exited `134` with `State isolation failure: multi-turn generation
diverged!`. Therefore no M29-0006 model-generation correctness pass or fresh
candidate A/B performance qualification is claimed. The prior M29-0005 4K/64K
timings remain baseline evidence only.

## N=2 same-MI50

Placement and validation remain metadata/ownership tests only. Production
attention integration is stopped at the contract boundary. The current
attention and KV-store launchers accept `kv_heads` but no source head stride or
head offset. They index Q/K/V as contiguous full-width tensors, so passing a
two-head shard directly would address the wrong heads, especially for shard 2.

Making this correct requires coordinated changes to the KV-store, prefill and
decode attention variants, query/output strides or offsets, and reduction
buffers. That is a broad kernel rewrite, so this workstream is classified:

`M29_5_TWO_SHARD_ATTENTION_BLOCKED_BROAD_KERNEL_REWRITE`

No broad kernel rewrite is included in M29-0006.

## Scope boundary

This record covers M29-0006 only. It does not claim M29.6 or M30 work, and it
does not claim a two-shard performance result.
