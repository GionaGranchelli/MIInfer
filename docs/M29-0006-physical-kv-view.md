# M29-0006 physical KV view

Status: `M29_5_N1_PHYSICAL_VIEW_CONTRACT_ONLY`

Contract base: `78adc52c85c05095e6cd0e5a034690b4df2b3b91`

Contract commit: `b71d08600c586927a7b4c07bbb3c7eef15a57d44`

N=1 integration commit: `5325fc49c992ed1884ff371d66e22147d26cd645`

## N=1

`PhysicalKvView` is now the explicit kernel-facing type at all four attention
entry points (forward, decode, and their profiled variants). `view()` still
resolves the legacy raw pointers once and populates the single shard with
`head_begin = 0` and `head_count = kKvHeads`. No per-element logical lookup,
allocation, copy, or synchronization was added.

The Z840 physical build/smoke result is pending because `192.168.68.54:22`
is currently timing out. The alternate Z840 endpoint `100.118.66.80` is
reachable and the contract test passes there, but this is not yet a production
placement integration qualification.

### A-side boundary

The current `PrefillV2Model` constructs `AttentionLayerKvCacheStorage` with
direct `hipMalloc` ownership and calls `storage.view()` at each layer. The
production model does not currently own or connect a `ContextSpace`,
`PlacementPlan`, or `DeviceKvPool`. Therefore the present change makes the
kernel-facing type explicit and proves the N=1 descriptor is pointer-equivalent
to the old view, but it does not yet satisfy the full placement-resolved chain.

The precise A-side classification is:

`M29_5_N1_PHYSICAL_VIEW_BLOCKED_PRODUCTION_PLACEMENT_NOT_WIRED`

Wiring those existing placement objects into production cache allocation and
retaining the current raw K/V layout requires more than the allowed tiny
interface adaptation; it is not claimed in this change.

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
