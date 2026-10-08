# M31-0005 — KV storage and memory ledger

Status: source ledger complete; runtime DRAM traffic and allocator peaks remain
measurement questions.

## Serving KV layout

For this Qwen3.8-27B runtime, the source dimensions are 16 full-attention
layers, 4 KV heads, head dimension 256, and FP16 K and V. `DeviceKvPool` owns
the cache and layer views refer to their assigned storage; the attention view
does not materialize a second cache. The unique logical K+V payload is
65,536 bytes per token:

| Context | Unique FP16 KV payload | Capacity allocation at context + 256 |
|---:|---:|---:|
| 8K | 512 MiB | 528 MiB |
| 32K | 2 GiB | 2.02 GiB |
| 64K | 4 GiB | 4.02 GiB |
| 128K | 8 GiB | 8.02 GiB |

The per-query-head logical load request upper bound is 6x unique KV because
24 Q heads share 4 KV heads. This does not imply 6x HBM reads: cache
coherency/reuse and actual DRAM counters were not measured.

## Attention scratch

Graph decode uses device-global Split-K arrays of approximately 1.51 MiB plus
12 KiB metadata. Eager suffix attention uses the workspace-manager allocation
with 512-token maximum width. For 32 splits it reserves about 387 MiB; for 3
splits about 36.3 MiB. These are separate paths and must not be combined in one
128K ledger. The existing capacity threshold chooses 3 eager splits above
66,000 tokens, saving about 350.6 MiB versus 32 splits.

## Persistent state and snapshots

Snapshots/forks are not part of the ordinary serving attention path. Existing
snapshot code stages K/V through host memory and accounts GDN state separately;
the configured snapshot budget and COW behavior must remain intact. This audit
found no evidence that ordinary graph decode duplicates the KV pool or that
M29/M30 snapshot ownership can safely be changed as a performance tweak.

The historical 128K MIInfer allocation report of 33.37 GB is a measured total
process/device footprint, not an 8 GiB KV estimate. The difference includes
weights, recurrent state, scratch, allocator reservations, and other runtime
allocations; a complete simultaneous allocation ledger is not available here.
Do not call the entire difference from llama.cpp waste.

## Precision decision

Q8 KV previously reduced 128K memory substantially (about 3.97 GiB in the
historical ledger), but a 64K experiment reported a latency regression. A
fused load/dequantization kernel could alter the trade-off but would need
reference parity, finite-value/boundary tests, and controlled timing. Q8 is not
recommended as a decode-speed optimization based on current evidence.
