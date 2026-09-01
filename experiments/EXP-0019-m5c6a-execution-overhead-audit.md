# EXP-0019 — M5-C6a execution-overhead attribution

**Status:** COMPLETE  
**Milestone:** M5  
**Date:** 2026-09-01  
**Candidate baseline:** `10433be7f476`

## Question

After persistent workspace allocation and resident normalization weights, what
accounts for the remaining fixed decode overhead? This is a measurement-only
slice; no production execution behavior is changed.

## Workload and source

The audit uses the clean C5b position-1 profile from
`bench/results/20260901T083350Z-372617/`, with the cooperative attention path,
Qwen3-8B Q4_0, and the MI50 Release build. The profile records deferred HIP
events and copy call sites. Its audit wall time is not used as a throughput
measurement.

## Per-token attribution

| Source | Calls/token | Bytes/token | Sync events/token | Dispatches/token |
|---|---:|---:|---:|---:|
| Layer input D2D copies | 36 | 589,824 | 36 | 0 |
| Layer output D2D copies | 36 | 589,824 | 36 | 0 |
| KV append K/V writes | 576 | 294,912 | 576 | 0 |
| Final logits D2H copy | 1 | 607,744 | 1 | 0 |
| **Copy-call subtotal** | **649** | **2,082,304** | **649** | **0** |
| Deferred profile finalization | 0 | 0 | 1 | 0 |
| **Measured total** | **649** | **2,082,304** | **650** | **1,588** |

The layer input/output counts are 36 layers per token. KV append performs one
K and one V copy for every cached head represented by the current layout. The
final logits copy is required by the current CPU-side greedy argmax path.

## Other fixed dispatch families

The same profile reports these unchanged dispatch families:

| Family | Dispatches/token |
|---|---:|
| Normalization | 289 |
| Quantization | 505 |
| Q/K/V projection | 108 |
| O projection | 36 |
| FFN projection | 108 |
| RoPE | 72 |
| Attention | 36 |
| Activation | 36 |
| Residual | 72 |
| Conversion | 324 |
| LM head | 1 |
| **Total** | **1,588** |

## Interpretation

KV append accounts for 576 of 650 measured synchronization events (88.6%),
but its writes are part of the persistent-cache contract and need a separate
K/V direct-write experiment to remove safely. The 72 layer handoff copies are
the clearest immediately avoidable materialization family: the fast path
already has stable ping-pong output ownership, so a candidate can test
writing the final residual directly to the next layer buffer. The final logits
copy is one call but transfers 607,744 bytes because greedy selection remains
host-side.

The dispatch table shows that quantization, normalization, conversion, and
projection launches remain substantial, while C5a/C5b have not changed the
1,588-launch topology. No optimization is accepted by this audit.

## Decision

```text
COMPLETE — overhead attribution recorded
```

## Follow-up

M5-C6b should isolate direct layer-output ownership by removing only the
36 fast-path layer-output D2D copies. Preserve the trace path and all kernel
precision boundaries, then run the same correctness and A/B benchmark gates.
K/V direct cache writes and dispatch/materialization fusion remain separate
experiments.
