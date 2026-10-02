# M29-0002 Workstream A — HIP VMM feasibility

## Decision

`HIP_VMM_VIABLE`

On the qualified Z840 MI50/gfx906 runtime, HIP virtual memory can reserve a
large virtual range, create and incrementally map device-backed allocations,
grant GPU read/write access, execute a gfx906 kernel against the mappings, and
unmap/release the mappings repeatedly.

## Tested

- `hipMemGetAllocationGranularity`: minimum granularity `4096` bytes.
- `hipMemAddressReserve`: `4294967296` bytes (4 GiB), aligned to the reported
  granularity.
- `hipMemCreate`, `hipMemMap`, and `hipMemSetAccess` for two incremental
  `65536`-byte chunks.
- GPU write/read correctness on both mappings after the second mapping was
  added; existing virtual addresses were unchanged.
- `hipMemUnmap`, `hipMemRelease`, and `hipMemAddressFree` lifecycle.
- 50 complete reserve/map/use/unmap/release cycles.

## Timings

One 50-cycle run reported:

```text
reserve_avg_us=12.86 map_avg_us=2880.44 unmap_release_avg_us=70.58
```

A second run reported:

```text
reserve_avg_us=11.24 map_avg_us=4895.32 unmap_release_avg_us=75.94
```

These operations are setup/lifecycle costs only; this spike does not put page
table lookup on a kernel hot path. The result supports using HIP VMM for
physical backing, subject to normal ROCm/gfx906 runtime support and future
fragmentation/performance measurement at the production allocation scale.

## Reproduction

```text
cmake --preset mi50-release
cmake --build --preset mi50-release --target miinfer-hip-vmm-feasibility -j2
./build/mi50-release/miinfer-hip-vmm-feasibility
```
