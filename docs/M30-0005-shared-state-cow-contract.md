# M30-0005 Shared Snapshot State and Copy-on-Write Contract

Status: frozen contract; implementation follows on `m30/shared-state-cow`.

## Scope

Snapshot state is process-local and memory-resident. M30-0005 does not add
disk persistence, remote state, multi-GPU placement, or a generic virtual-memory
layer. The existing `PrefillV2Model` snapshot API remains the authority.

## Ownership model

Each snapshot ID owns one reference to an immutable `SnapshotBacking` node.
Forks and exact-prefix snapshots add references to the same node. A node may
retain one parent node and only its divergent suffix GQA bytes; its GDN state
is a private immutable successor because recurrent state mutates as a whole at
continuation boundaries. Parent nodes remain alive until their final record or
child reference is released.

```text
Snapshot A ─┐
Snapshot B ─┼── immutable backing P (shared GQA prefix + GDN state)
Snapshot C ─┘
                    │
                    └── backing P+A (private GDN + divergent GQA suffix)
```

Restoring reconstructs the active state by walking the backing chain from the
oldest prefix to the selected node, uploads each retained GQA range, and
restores the selected node's GDN state. Continuation mutates only the live
model state; immutable snapshot backing is never modified.

## API and limits

The M30-0004 names remain:

```text
SnapshotId snapshot(prefix_tokens)
SnapshotId fork(snapshot_id)
bool restore_snapshot(snapshot_id)
bool rollback_snapshot(snapshot_id)
bool release_snapshot(snapshot_id)
void clear_snapshots()
```

Snapshot IDs are process-local, monotonic, and zero is invalid. The existing
maximum of eight logical snapshots and 3 GiB of unique physical backing is
preserved. Invalid, released, evicted, reset, or session-local IDs return
failure without mutating active state.

## Accounting

The runtime exposes:

- logical snapshot count;
- unique physical shared bytes;
- unique private branch bytes;
- backing reference count;
- COW event count;
- bytes copied into divergent backing nodes.

Physical bytes are counted once per immutable backing node, not once per
logical snapshot ID.

## Required invariants

- Multiple IDs from one prefix share one physical backing.
- A continuation or successor snapshot never mutates a sibling backing.
- Fork/restore/rollback/nested/repeated operations preserve M30-0004 parity.
- Releasing records and resetting/destroying a session releases all backing.
- Zero-suffix restore and independent-session rejection remain valid.
- No snapshot state is written to disk by the new API.

