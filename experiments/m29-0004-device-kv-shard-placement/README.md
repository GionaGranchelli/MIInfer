# M29-0004 Workstream B — DeviceKvShard + PlacementPlan

Status: `M29_3_DEVICE_KV_SHARD_PLACEMENT_QUALIFIED`

Base: `31ae821d2c04184fe81ff4441977c5556f7a2b55`

The placement layer consumes the frozen `ContextSpace` contract. A shard has
one immutable owning device, logical page/range metadata, optional KV-head
metadata, and a physical range/view. `PlacementPlan` validates committed page
boundaries, rejects overlapping logical placement and overlapping ranges on the
same physical backing, resolves complete N=1 ranges, and replaces physical
metadata without changing logical page identity.

No `DeviceKvPool`, VMM, kernel, hot-path page lookup, multi-device execution,
two-shard execution, COW, snapshot, or M30 behavior is included.

## Evidence

- `cmake --preset host-only`
- `cmake --build build/host-only --target miinfer-device-kv-shard-placement-test -j2`
- `ctest --test-dir build/host-only -R device-kv-shard-placement-host --output-on-failure`
- focused test covers N=1 resolution, identity-preserving replacement, owner
  metadata, invalid/missing/overlapping placement, and bounds/errors;
- physical preflight is separate and must use `HIP_VISIBLE_DEVICES=1`, verify
  `gfx906`, and verify the approximately 32 GiB device before GPU validation.

The implementation is placement metadata only; no physical allocation is
performed by this workstream.
