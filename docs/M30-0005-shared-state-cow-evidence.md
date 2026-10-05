# M30-0005 Shared Snapshot State and COW Evidence

## Contract and implementation

```text
M30_0005_BASE_SHA=d1d6327232f9270a301867a396d3fbe067ed6b60
M30_0005_CONTRACT_SHA=f071bb26b0f7950704dc6ce169570577667b3bf7
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

| Metric | M30-0004 full-copy | M30-0005 COW |
|---|---:|---:|
| Branch operations | 8 | 8 |
| Snapshot count | 8 | 8 |
| Physical checkpoint bytes | 1,275,332,088 | 477,233,152 |
| Shared bytes | n/a | 318,308,352 |
| Private bytes | 1,275,332,088 | 158,924,800 |
| Reference count | n/a | 10 |
| COW events | n/a | 2 |
| COW bytes copied | n/a | 317,849,600 |
| Peak VRAM | 25,202,993,152 | 25,680,226,304 |
| Branch wall time | 4,239.35 ms | 657.22 ms |
| Physical suffix tokens | 6 | 6 |
| Cold physical tokens | 48 | 48 |

Physical checkpoint memory decreased by 798,098,936 bytes (62.6%) at the same
branch depth. All representative COW outputs matched cold outputs exactly.

The peak-VRAM comparison needs an explicit qualification: the M30-0004 control
stores full copies as disk-backed `.miinfer` files, while M30-0005 intentionally
forbids disk persistence and keeps shared backing on the GPU. Therefore COW
uses 477,233,152 more VRAM than that disk-backed control even though its unique
checkpoint memory is 62.6% lower. This is the expected consequence of the
no-disk requirement, not a claimed VRAM win over disk.

## Gate result

- shared prefix backing: PASS;
- reference counting/lifetime cleanup: PASS;
- COW isolation and sibling integrity: PASS;
- rollback/cold parity: PASS;
- physical checkpoint memory reduction: PASS, 62.6%;
- useful branch depth at the tested topology: PASS, eight snapshots;
- VRAM reduction versus the disk-backed M30-0004 control: NOT APPLICABLE;
- disk persistence in the new snapshot API: PASS, none.

The remaining product decision is whether device-resident no-disk snapshots
should be compared against a future device-resident full-copy control for a
strict VRAM gate. That control is outside the M30-0004 implementation and was
not substituted silently here.

