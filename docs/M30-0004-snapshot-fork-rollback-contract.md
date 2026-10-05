# M30-0004 Snapshot, Fork, and Rollback Contract

Status: frozen contract; implementation follows on `m30/snapshot-fork-rollback`.

## API

`PrefillV2Model` exposes one process-local snapshot namespace:

```text
SnapshotId snapshot(prefix_tokens)
SnapshotId fork(snapshot_id)
bool restore(snapshot_id)
bool rollback(snapshot_id)
bool release_snapshot(snapshot_id)
void clear_snapshots()
```

The C++ names are `snapshot`, `fork`, `restore_snapshot`,
`rollback_snapshot`, `release_snapshot`, and `clear_snapshots`. `SnapshotId`
is an opaque monotonic integer; zero is invalid.

## State contract

The caller supplies the logical token sequence represented by the currently
active model state. `snapshot()` captures the active full state through the
existing persistent-session representation: all 16 GQA KV caches, all 48 GDN
states, model/quantization identity, token hash, and prefix position.

Snapshot IDs are process-local and resolve to durable files owned by the model.
They do not depend on request IDs, raw pointers, or mutable current-session
state. `fork()` creates an independent full-state copy, so later restore,
release, and eviction of one ID do not invalidate another.

`restore_snapshot()` and `rollback_snapshot()` select the snapshot state,
restore both GQA and GDN state, and make suffix-only continuation immediately
possible. They do not replay the represented prefix. Rollback is an explicit
semantic alias for restore in this first implementation; newer active state is
discarded by the restore.

The first implementation is bounded to the existing eight-checkpoint / 3 GiB
full-state budget. Invalid, released, evicted, reset, or session-local IDs
return `false` and do not mutate active model state. Restoring a snapshot with
no subsequent suffix is a valid zero-token restore operation.

## Required invariants

- Snapshot → continue → rollback → continue produces the same result as a
  direct continuation from the snapshot.
- Two forks from one snapshot are independent.
- Repeated rollback and nested snapshots remain valid until release/eviction.
- Session reset and destruction release all snapshot references and files.
- Snapshot operations never introduce COW, shared pages, or rollback-specific
  execution authorities.

