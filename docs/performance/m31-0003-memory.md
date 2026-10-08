# M31-0003 — VRAM ownership and accounting

## Source-derived model

The M31 checkpointed path constructs `PrefillV2Model` with context capacity
equal to prompt length plus 256. Sixteen attention layers each allocate their
own KV pool. FP16 K+V at 4 KV heads × 256 dimensions × 2 bytes is 65,536 bytes
per token per attention layer.

| Context | Model-owned memory (existing ledger) | Main context-dependent growth |
|---:|---:|---|
| 8K | 25.690 GB | KV + active GDN state |
| 32K | 27.300 GB | KV + active GDN state |
| 64K | 29.448 GB | KV + active GDN state |
| 128K | 33.375 GB | source-derived/historical; exceeds a 32 GB MI50 |

The table is not a fresh device peak measurement. The existing ledger also
records weights of 24,068,487,168 bytes, workspace of 727,711,744 bytes at
8/32/64K and 359,956,480 bytes at 128K, and a GDN cached checkpoint of
158,859,264 bytes. The 128K workspace changes with split selection. The
8K `hipMemGetInfo` residual of roughly 0.88–0.96 GB remains unclassified;
pre/post free-memory samples are not peak accounting.

## Implemented safe reclamation

`GdnCheckpointStorage` previously allocated 158,859,264 bytes in its
constructor, even for one-shot requests that never save a prefix. It now
allocates on the first successful checkpoint capture. `memory_bytes()` tracks
actual allocated checkpoint capacity: zero before first save, 158,859,264 after
save. `clear()` still invalidates but retains that allocation, preserving reuse
and avoiding repeated allocator churn. Partial allocation failure frees the
first buffer before the HIP error is surfaced.

For no-prefix-cache operation, expected model-owned memory is therefore lower
by 158,859,264 bytes (~151.5 MiB), subject to the existing ledger's other
categories. This is a statically known capacity reduction; actual usable VRAM
and temperature impact require hardware validation. Prefix reuse retains the
same buffer and capture/restore semantics.

## Remaining footprint gap

MIInfer's 24.07 GB resident weights versus llama.cpp's 16.81 GB is a cross-host
historical comparison and not fully matched. The additional 7.13 GB fused
gate/up layout is close in scale to the roughly 7.26 GB 8K weight difference,
but it coexists with MMQ gate/up weights and may serve a distinct hot path; it
cannot safely be removed based on byte similarity alone.

M30 snapshot capture has separate COW costs: KV is shared, while GDN state is
copied. A full 128K snapshot is estimated around 8.6 GB plus small hidden state
and may not fit after the base model nearly fills VRAM. Snapshot storage is not
part of the cold inference ledger.

## Offline validation and limit

`miinfer-reusable-context-allocation-test` uses mocked HIP calls to verify
zero checkpoint allocation before save, exactly two allocations on first
save, expected byte accounting, 96 state/history copies, valid capture, and
clear-retains-capacity behavior. It does not validate actual HIP allocation,
stream ordering, or GPU state parity.
