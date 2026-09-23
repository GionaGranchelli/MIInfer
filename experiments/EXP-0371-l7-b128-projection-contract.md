# EXP-0371 — Isolate L7 B128 projection contract

## Question

Does the L7 B128 partial-tail divergence begin in L7 input/normalization or in
the Q/K/V projection/preparation path?

## EXP-0370 evidence

EXP-0370 found large L7 projected K/V differences before KV storage, while
stored-cache prefix `[0,1536)` and the actual per-query causal bounds remained
correct. The next comparison therefore moves to the common L7 input hidden
before attention RMS normalization.

## L7 input-hidden comparison

The same deterministic P1664 control/candidate pair captured four tail rows at
the exact L7 input boundary. The candidate uses the existing B128 preparation
route; control uses scalar `run()` for the same positions.

| Position | Max abs | Mean abs | RMS |
|---:|---:|---:|---:|
| 1536 | `6.46238` | `0.0797554` | `0.144933` |
| 1537 | `6.19130` | `0.107159` | `0.170555` |
| 1538 | `3.99978` | `0.0841267` | `0.131501` |
| 1539 | `5.05482` | `0.105149` | `0.166359` |

This is the first captured unique divergence. It occurs before L7 attention
RMS normalization and before any Q/K/V projection.

## Downstream captures

For completeness, downstream candidate/control differences were also measured,
but they are not origin evidence because L7 input already differs:

| Boundary | Position 1536 max abs | Position 1539 max abs |
|---|---:|---:|
| normalized input | `2.83147` | `6.64022` |
| raw Q | `2.08116` | `3.32551` |
| raw K | `1.66575` | `9.51574` |
| gate | `2.87075` | `3.98053` |
| raw V | `2.50350` | `3.21989` |

The V result is especially useful diagnostically: its divergence does not
involve K normalization or RoPE, but it is downstream of the already-divergent
L7 input and therefore does not establish a V projection defect.

## Exact routes

| Route | L7 input source | Normalization/projection route |
|---|---|---|
| control | scalar `run()` input row | scalar per-token path; Q/K/V are produced from `normalized_input` |
| candidate | layer-major `chunk_input + local row` | `prepare_prefill_batch` B128 preparation; batched Q/K and V preparation |

The candidate capture used `inputs`, `prefill_normalized`, `prefill_qfull`, and
`prefill_value` for the active B128 rows. The control capture used `input`,
`normalized_input`, `qfull_dest`, and `value_input` at positions 1536--1539.
No projection geometry, quantization, or kernel was changed.

## Buffer/index audit status

The active capture paths use the source row pointers and the established
strides: L7 input rows are `kHidden`, combined Q/K rows are `12288+1024`, and
V rows are `1024`. The capture proves the semantic tensor mismatch but does not
prove whether the cause is stale L6 output, inter-layer buffer selection, or a
wrong B128 source row. No sentinel or device-address instrumentation was
needed after the first unique boundary was found.

## Decision

**LEARN** — the first unique divergence is L7 input hidden, before attention
normalization and projection. The raw K/V differences are downstream effects;
no K-specific normalization/RoPE or KV-store correction is justified. No
local fix was made.

The existing EXP-0367 P640 continuation remains exact for eight tokens. No new
P1664 continuation or timing was run because the protocol requires stopping at
the first unique semantic divergence.

## Next PRIMARY frontier

Qualify the L6→L7 inter-layer output contract for the B128 partial tail at
positions 1536--1539. Capture L6 output, the L7 input source row, and any
fused inter-layer normalization boundary before revisiting Q/K/V projections.

No new math kernel or unrelated performance optimization was implemented.
