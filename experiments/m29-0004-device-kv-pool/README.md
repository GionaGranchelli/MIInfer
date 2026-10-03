# M29-0004 Workstream A — DeviceKvPool

Status: `M29_3_DEVICE_KV_POOL_QUALIFIED`

Base: `31ae821d2c04184fe81ff4441977c5556f7a2b55`

`DeviceKvPool` is a device-local owner for one contiguous HIP allocation. It
uses the existing gfx906 validation path, exposes explicit capacity,
committed/available accounting, allocates aligned subranges, releases and
coalesces them, rejects foreign or repeated releases, and frees the backing
allocation deterministically. It is not a general allocator and does not
expose HIP-VMM handles.

## Evidence

- `cmake --preset mi50-release`
- `cmake --build --preset mi50-release --target miinfer-device-kv-pool-test -j2`
- `HIP_VISIBLE_DEVICES=0 ./build/mi50-release/miinfer-device-kv-pool-test`
- physical test passed on `gfx906:sramecc+:xnack-` with a 32 GiB-class device;
- focused test covers ownership, capacity/accounting, aligned allocation,
  exhaustion, release/reuse, repeated release rejection, and cleanup.

No `DeviceKvShard`, `PlacementPlan`, multi-device allocation, VMM backing,
COW, snapshots, or M30 behavior is included.
