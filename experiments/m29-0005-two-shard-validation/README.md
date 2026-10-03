# M29-0005 Workstream B — adversarial two-shard validation

Status: `M29_4_TWO_SHARD_INVARIANTS_QUALIFIED`

Base: `M29_0005_BASE_SHA=06c457567f466749d6ccc6180c88bf73b916fe85`.

The test explicitly selects visible device 0 after running with
`HIP_VISIBLE_DEVICES=1`; preflight verifies `gfx906` and 34.34 GB VRAM on the
Machinist MI50. It exercises a 128K logical ContextSpace with 32 pages,
one-shard placement, two disjoint head shards per page, and cleanup/reuse.

The two-shard payload remains equal to the one-shard payload: 262,144 bytes
committed in either representation. The test checks same-device ownership,
complete head coverage, stable logical IDs, duplicate/missing coverage,
released allocation ownership, invalid owner metadata, safety reserve, and
deterministic cleanup. The physical placement layer remains metadata-only;
attention kernels, VMM, peer access, and M30 are untouched.
