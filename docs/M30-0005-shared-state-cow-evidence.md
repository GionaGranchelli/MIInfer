# M30-0005 Shared Snapshot State and COW Evidence

## Contract and implementation

```text
M30_0005_BASE_SHA=d1d6327232f9270a301867a396d3fbe067ed6b60
M30_0005_CONTRACT_SHA=f071bb26b0f7950704dc6ce169570577667b3bf7
COW_FINAL_SHA=adafdf5d1226052f4bfccb5d6e7c388ed180c520
WORKLOAD_FINAL_SHA=e7778931f4cbdb27e28e70dc8841ed5ef57de901
INTEGRATED_SHA=5ad01528cbbd78e617628687758d74c1d2a8a4eb
```

The production `PrefillV2Model` snapshot API now uses immutable, device-resident
backing nodes. Forks and exact-prefix snapshots share one node by reference.
Successor snapshots retain the parent GQA prefix range, copy only the divergent
GQA suffix, and store a private GDN successor state. Snapshot restore walks the
backing chain and reconstructs the live GQA/GDN state without disk I/O.

The runtime reports logical count, unique shared/private bytes, references, COW
events, and COW bytes copied. Reset, release, eviction, destruction, and
independent-session behavior remain covered by the M30-0004 lifecycle test.

## Workload

Host: Machinist, `HIP_VISIBLE_DEVICES=1`, Qwen3.8-27B-Q4_K_M. The topology
contains a shared base, four sibling forks, a divergent branch-A snapshot, a
nested fork, and a nested successor snapshot: eight logical snapshots total.
Representative branch outputs are compared against cold execution.

| Metric | Device full-copy control | M30-0005 COW |
|---|---:|---:|
| Branch operations | 8 | 8 |
| Snapshot count | 8 | 8 |
| Physical checkpoint bytes | 1,275,330,560 | 477,233,152 |
| Shared bytes | n/a | 318,308,352 |
| Private bytes | 1,275,330,560 | 158,924,800 |
| Reference count | n/a | 10 |
| COW events | n/a | 2 |
| COW bytes copied | n/a | 317,849,600 |
| Peak VRAM | 26,478,323,712 | 25,680,226,304 |
| Restore latency, eight restores | 19.59 ms | 24.74 ms |
| Branch wall time | 656.79 ms | 657.42 ms |
| Physical suffix tokens | 6 | 6 |
| Cold physical tokens | 48 | 48 |

Physical checkpoint memory decreased by 798,097,408 bytes (62.6%) at the same
branch depth. Peak VRAM decreased by the same 798,097,408 bytes (3.0% of the
device full-copy run). All representative COW outputs matched cold outputs
exactly.

The primary table uses a fair device-resident full-copy control selected with
the internal `MIINFER_M30_0005_FULL_COPY=1` switch; production defaults remain
COW. The original M30-0004 disk-backed control also passed the same workload:
1,275,332,088 checkpoint bytes, 25,202,993,152 peak VRAM, 1,345.19 ms restore
latency, and 4,241.86 ms branch wall time. It is retained as historical
compatibility evidence, not used for the VRAM comparison.

## Gate result

- shared prefix backing: PASS;
- reference counting/lifetime cleanup: PASS;
- COW isolation and sibling integrity: PASS;
- rollback/cold parity: PASS;
- restore latency accounting: PASS, 24.74 ms COW vs 19.59 ms device full-copy;
- physical checkpoint memory reduction: PASS, 62.6%;
- peak VRAM reduction at equal depth: PASS, 3.0%;
- useful branch depth at the tested topology: PASS, eight snapshots;
- VRAM reduction versus the disk-backed M30-0004 control: NOT APPLICABLE;
- disk persistence in the new snapshot API: PASS, none.

The device full-copy control is test-only and exists solely to make the VRAM
gate apples-to-apples; it is not a second production cache authority.
