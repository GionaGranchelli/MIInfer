# M30-0004 Snapshot, Fork, and Rollback Evidence

## Delivered

`PrefillV2Model` now exposes a process-local `SnapshotId` API:

- `snapshot(tokens)` captures the full existing session representation;
- `fork(id)` copies an independent full checkpoint;
- `restore_snapshot(id)` and `rollback_snapshot(id)` restore GQA KV and GDN;
- `release_snapshot(id)` and `clear_snapshots()` remove files and references.

Snapshots use the existing persistent-session format, are bounded to eight
entries and 3 GiB, and are owned by one model instance. Reset, destruction,
release, and LRU eviction remove their files. Restore clears live KV storage
before upload and consumes the loaded full state directly for the next suffix;
this prevents stale tail KV and duplicate GDN restores across repeated branch
retries.

## Workstream A — lifecycle gate

Host: Machinist, `HIP_VISIBLE_DEVICES=1`, Qwen3.8-27B-Q4_K_M.

The final HIP executable passed:

```text
M30-0004 snapshot/fork/rollback: PASS
snapshot=PASS fork=PASS rollback=PASS nested=PASS invalid=PASS
release=PASS eviction=PASS reset=PASS cold_parity=PASS
```

The test covers snapshot/continue, zero-suffix restore, rollback, two-way
forking, nested and repeated rollback, invalid/released IDs, nine-entry LRU
eviction, reset invalidation, independent model lifetime, destruction cleanup,
and deterministic cold replay parity.

## Workstream B — branch/retry workload

The final workload ran ten deterministic branch/retry operations on the same
Machinist MI50 production `PrefillV2Model` engine used by `miinfer serve`.
Each candidate operation restored one base snapshot and executed only its
divergent suffix; the control reconstructed the branch with cold replay.

| Metric | Result |
|---|---:|
| Operations | 10 |
| Replay physical prompt tokens | 103 |
| Restored-suffix physical tokens | 23 |
| Tokens avoided | 80 (77.7%) |
| Candidate wall time | 3502.35 ms |
| Cold/replay wall time | 4881.72 ms |
| Wall-time improvement | 1379.37 ms (28.3%) |
| Checkpoint bytes | 159,383,741 |
| Total model VRAM | 25,202,993,152 bytes |

All ten generated-token results were exactly equal between candidate and cold
control. An earlier workload using arbitrary high token IDs exposed a mismatch;
the workload was corrected to use the repository's existing qualified branch
tokens, and the final ten-operation run passed. The high-prefix arbitrary-token
screen is not claimed as qualification.

## Qualification boundary

This milestone adds the production engine API and validates it through the
same `PrefillV2Model` path used by serving. It does not add an HTTP snapshot
endpoint or COW/shared-page storage. Full checkpoint copies remain the
intentional first implementation; the measured checkpoint is approximately
159 MiB and the existing eight-entry/3 GiB bound is unchanged.

```text
M30_0004_CONTRACT_SHA=a9b763d0ab3804fc1c3a3aacdd32076b1e58717a
```
