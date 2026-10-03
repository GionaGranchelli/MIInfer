# M29-0006 physical KV view

Status: `M29_5_N1_PHYSICAL_VIEW_INTEGRATED_GENERATION_PASS_PERF_PENDING`

Contract base: `78adc52c85c05095e6cd0e5a034690b4df2b3b91`

Contract commit: `b71d08600c586927a7b4c07bbb3c7eef15a57d44`

N=1 integration commit: `024f1c2`

## N=1

`PhysicalKvView` is now the explicit kernel-facing type at all four attention
entry points (forward, decode, and their profiled variants). Production
`AttentionLayerKvCacheStorage` allocates one placement-owned `DeviceKvPool`
block, registers a full-capacity `ContextSpace` page and `PlacementPlan` shard
for all KV heads, and caches the resulting raw-pointer view. No per-element
logical lookup, allocation, copy, or synchronization was added.

The Z840 alternate endpoint `100.118.66.80` is reachable. The physical-view,
HIP smoke, ContextSpace, and placement tests pass there. The model-level
PrefillV2 harness now passes state isolation, 16-step decode continuity, and
generation for P64/P512/P2048 after the minimal deterministic RNG reset fix in
`generate()`. The same harness failed on the M29-0005 base, so that fix is
independently attributable to the pre-existing harness defect.

The final candidate rerun measured P64: 550.22 ms TTFT / 34.00 ms decode
step, P512: 2071.88 ms / 34.69 ms, and P2048: 8538.58 ms / 35.87 ms.

Fresh integrated 4K/64K paired A/B measurements remain pending. The prior
M29-0005 4K/64K timings remain baseline evidence only until rerun against this
placement-backed candidate.

The final candidate context envelope passed 32K, 64K, and 128K construction:
128K used 31.07 GiB and retained 0.11 GiB observed device free memory. A
candidate 128K prefill/decode/generation run was not repeated because the
existing focused run takes tens of minutes; the retained M29-0003 result is
baseline evidence, not a new M29-0006 execution claim.

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
