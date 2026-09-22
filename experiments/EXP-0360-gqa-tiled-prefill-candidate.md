# EXP-0360 — GQA-shared tiled full-attention prefill candidate

## Hypothesis

For Qwen3.8-27B full-attention prefill, sharing BK32 F16 K/V tiles across the
six query heads in each GQA group will reduce repeated KV reads at long
context.

## Baseline

The existing `qwen35_tiled_online_attention_batch_f16_kernel` uses one Wave64
per `(token, query_head)` and scans the causal KV prefix independently.

## Candidate

`qwen35_gqa_tiled_online_attention_batch_f16_kernel` uses one 384-thread
workgroup per `(token, KV head)`: six Wave64 waves compute the six associated
query heads and share one BK32 K/V tile in LDS. It is selected only by
`MIINFER_PREFILL_GQA_TILED_ATTN=1`, with the existing path retained as control.

## Correctness

The existing GPU batch test now exercises 24 query heads, 4 KV heads,
head_dim 256, nonzero base position, and a partial BK32 tail. The candidate is
compared with the independent CPU attention calculation using the production
KV-head-major layout.

Result: `build/mi50-release/miinfer-qwen35-conv-batch-test` passed.

## Build

```text
cmake --build --preset mi50-release --target miinfer miinfer-qwen3-primitives-test -j2
cmake --build --preset mi50-release --target miinfer-qwen35-conv-batch-test -j2
```

Both builds passed for gfx906.

## Performance

Standalone GPU-event benchmark, five measured iterations after two warmups,
random exact-shape tensors, same production KV-head-major layout:

| Tokens | Current ms | Candidate ms | Candidate speedup | Max abs error |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 2.586 | 4.583 | 0.564x | 0 |
| 2048 | 40.428 | 79.060 | 0.511x | 0 |
| 4096 | 177.494 | 313.777 | 0.566x | 0 |
| 8192 | 749.019 | 1238.090 | 0.605x | 0 |
| 16384 | 3274.710 | 4874.590 | 0.672x | 0 |

The candidate fails the immediate P512 gate (>3% regression) and is slower at
all tested contexts. No end-to-end promotion or thermal claim is justified.

## Forensic P4K profile

The standalone GPU-event run at 4096 tokens measured `177.494 ms` for the
control and `313.777 ms` for the candidate, with zero output error. The
available host has no `rocprof`, `rocprofv2`, or Omniperf installation, so
hardware VMEM/L2/HBM/VALU counters and measured VGPR usage cannot be claimed.

Static execution facts do establish the synchronization/resource tradeoff:

| Property | Control | Candidate |
| --- | --- | --- |
| Workgroup | 64 threads / 1 Wave64 | 384 threads / 6 Wave64 |
| Dynamic LDS | 0 | 32768 bytes |
| Tile barriers | 0 | 2 per BK32 tile |
| P4K tile barriers per workgroup | 0 | 256 |
| K/V staging | independent wave loads | one LDS tile per KV-head group |

Thus the result supports, but does not counter-prove with hardware counters,
the narrower conclusion: explicit GQA reuse through six-wave cooperative LDS
tiling is a losing strategy on gfx906. It does not show that GQA reuse itself
has no value; the improving speed ratios at longer contexts are consistent
with some reuse benefit being present but outweighed by synchronization and
resource cost.

## Decision

REJECT. Explicit GQA reuse through six-wave cooperative LDS tiling is a losing
strategy on gfx906 for this workload. Candidate 2, layered on this topology,
is not justified by the Candidate 1 gate.
