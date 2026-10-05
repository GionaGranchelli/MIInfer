# M29-0005 Workstream A — two-shard single-MI50 placement

Status: `M29_4_TWO_SHARD_PLACEMENT_QUALIFIED`

Base: `M29_0005_BASE_SHA=06c457567f466749d6ccc6180c88bf73b916fe85`.

The existing placement model now permits multiple shards for one logical page
only when their KV-head ranges are non-empty and disjoint. The qualification
uses one `ContextSpace` page and one Z840 `gfx906` MI50. It transitions
one-shard → two-shard → one-shard while keeping the same `LogicalPageId`.

Physical accounting uses one 8 KiB one-shard allocation or two 4 KiB pool
allocations. The two-shard payload is therefore 8 KiB total, with no payload
duplication; each shard has one owner and disjoint physical storage. The test
also rejects duplicate head coverage, rejects incomplete required head
coverage, verifies reallocation, and returns pool accounting to zero on
cleanup.

No attention/kernel integration, multi-GPU, peer access, VMM, or M30 behavior
is included.
