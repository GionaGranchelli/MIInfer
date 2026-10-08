# M31-0006 — KV and scratch accounting

## Direct attention microbenchmark

The measured shape uses 4 KV heads, D=256, FP16 K and V, and capacity
`context + 1` (the extra slot is poisoned). The source-derived K+V sizes and
observed HIP allocation deltas are:

| Context | K+V bytes | Observed allocation delta | Difference beyond K+V |
|---:|---:|---:|---:|
| 8K | 33,558,528 | 192,937,984 | 159,379,456 |
| 32K | 134,221,824 | 293,601,280 | 159,379,456 |
| 64K | 268,439,552 | 427,819,008 | 159,379,456 |
| 128K | 536,875,008 | 696,254,464 | 159,379,456 |

The identical difference is a runtime/module/allocation floor, not yet
decomposed into its owners. It must not be called scratch. Live allocation
including the pre-existing HIP context ranged from 213,909,504 B at 8K to
717,225,984 B at 128K. The 128K shape left over 33.6 GB reported free on the
32-GiB-class device; it is not representative of full-model weights or full
model KV residency.

The kernel's static split scratch is
`64 * 24 * 256 * sizeof(float) + 2 * 64 * 24 * sizeof(float)` =
1,585,152 B. It remains allocated at 16, 32, or 64 active splits; fewer active
splits reduce work/reduction width but do not resize these arrays. Observed
VRAM allocation and static scratch therefore answer different questions.

## Model-level traffic derivation

For the 16 full-attention blocks in this model, each position contains FP16 K
and V for four KV heads of width 256:

`16 * 4 * 256 * 2 tensors * 2 bytes = 65,536 bytes/position` (64 KiB).

At 128K positions that is 8 GiB of unique K+V state. If every one of the 24
query heads independently reloads its mapped KV head for every decode token,
the logical load upper bound is six times larger, 48 GiB/token at 128K. GQA
reuse in cache may lower external-memory traffic; this is a source/model
derivation, not a hardware-counter measurement. No DRAM byte counters were
collected.

The large gap between the 8-GiB unique state and the possible repeated-read
bound makes KV reuse/coalescing more promising than reducing active splits,
but instrumentation is needed before changing code. Do not infer actual
bandwidth from the logical upper bound.

Raw observed ledgers are in the baseline result logs linked from
[graph-decode-baseline](m31-0006-graph-decode-baseline.md).
