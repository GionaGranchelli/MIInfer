# M29-0003 Workstream B — ContextSpace + logical pages

Status: `M29_2_CONTEXTSPACE_FOUNDATION_QUALIFIED`

## Authority

- Base: `M29_0003_BASE_SHA=072072874b63344ed8884333fd135c52411038ae`
- Branch: `m29/context-space`
- Host: Machinist X99 / MI50
- GPU preflight: `HIP_VISIBLE_DEVICES=1`; selected device is `gfx906` with
  `34342961152` bytes reported VRAM (32 GiB class). The unselected device is
  `gfx1200` with approximately 16 GiB and was not used.

## Implemented substrate

- `ContextSpace` owns logical capacity, committed extent, and logical pages.
- Pages receive monotonic `LogicalPageId` values independent of physical
  placement.
- `resolve(LogicalRange)` is the explicit boundary that returns page metadata
  plus a `PhysicalPageView`; no logical lookup is introduced into kernels.
- `remap()` changes only the physical view and preserves the logical identity.
- The implementation contains no HIP, VMM, device pointer, GPU ID, pool,
  shard, COW, snapshot, or multi-GPU dependency.

## Evidence

Host-only Release build and focused CTest:

```text
cmake --build build/qualification-host --target miinfer-context-space-test -j2
[100%] Built target miinfer-context-space-test
ctest --test-dir build/qualification-host -R context-space-host --output-on-failure
1/1 Test #6: context-space-host ... Passed
100% tests passed, 0 tests failed out of 1
```

The test covers empty state, capacity, append/growth, page boundaries,
multiple pages, stable identity, physical remapping, unresolved pages,
bounds/errors, and scoped cleanup. It uses explicit runtime checks so the
coverage remains active in Release builds.

No physical HIP integration test was added: this workstream is intentionally
placement-independent, and the required invariants are fully host-testable.
The existing MI50 preflight was run separately before any GPU validation.

## Deliberate non-goals

No production attention-path integration, HIP-VMM coupling, generic allocator,
`DeviceKvPool`, `DeviceKvShard`, COW, snapshots, fork/rollback, shared-prefix
ownership, refcounting, distributed placement, or M30 work is included.
