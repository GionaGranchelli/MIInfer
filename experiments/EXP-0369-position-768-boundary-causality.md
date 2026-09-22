# EXP-0369 — Explain the P1664 position-768 divergence

## Question

Why did the prior P1664 comparison appear to diverge at position 768 when the
actual partial tail begins at position 1536?

## Same-route reproducibility

Two control runs and two candidate runs used the same P1664 prompt, model,
context capacity, and qualified preset. State snapshots were compared by
layer, including recurrent state/history and active attention K/V.

| Pair | Result |
|---|---|
| control A vs control B | byte-identical state records |
| candidate A vs candidate B | byte-identical state records |

The oracle is deterministic enough for candidate/control comparison.

## Route trace

An opt-in `MIINFER_EXP0369_TRACE_ROUTE=1` trace records per-layer chunk base,
count, absolute base, preparation, deferred-tail, and batched-attention
decisions. It does not alter the default route. Both routes trace full chunks
at absolute bases `0`, `512`, and `1024`, each with count `512`.

At the final chunk:

| Route | Absolute base | Count | Prepared | Batched attention |
|---|---:|---:|---:|---:|
| control | 1536 | 128 | false | false |
| candidate (`MIINFER_EXP0366_PARTIAL_TAIL=1`) | 1536 | 128 | true | true |

H1 is proven: the selector changes execution only at the actual partial tail.
There is no traced internal 256-token boundary at 768. H2 is rejected.

## Corrected state comparison

The state serializer writes each attention head as all K positions followed by
all V positions, with 256 values per position. The prior comparison treated
the flattened K block as if K and V were interleaved, producing the false
mapping `1536 * 256 -> position 768`.

With the correct mapping:

| Field | Result |
|---|---|
| last identical L7 K/V prefix | `[0,1536)` |
| first divergent layer | attention L7 |
| first divergent tensor | K |
| first divergent position/dimension | position 1536, dimension 0 |
| first delta | `0.0078125` |
| later L7 K/V maxima | `0.111328` / `0.111328` |

The candidate/control state pairs are deterministic, so H4 is rejected. The
serialization/comparison artifact explains the reported 768, so H3 is proven
for that observation. H5 is not established.

## Position and causal audit

The route trace supplies outer base `1536`, inner chunk base `0`, count `128`,
and local indices `[0,127]`. The source passes `base_position + token` to the
batched RoPE and KV-store kernels, so the first intended values are:

```text
absolute position = 1536 + 0 = 1536
RoPE position     = 1536
KV write position = 1536
```

The batched attention API receives the same base and count. Per-query causal
read length is not emitted by the current trace, so an actual first/last
readable-KV capture remains open. No prefix mutation was observed: L7 K/V
positions `[0,1536)` are identical.

## L7 intermediate-stage status

No intermediate tensor capture was needed to explain the false position 768.
The first state-level difference is the candidate's legitimate route boundary
at L7 K/V position 1536. Projection, normalization, post-RoPE, and pre-write
values were not captured, so this record does not claim a narrower stage defect
or prove the batched causal read bound.

## Decision

**LEARN** — position 768 was a serializer/comparison indexing artifact. The
candidate/control routes are identical through 1536 and first differ because
the opt-in selector enables batched partial attention for the actual B128
tail. No local correctness fix was made and no promotion is authorized until
the L7 partial-tail stage and causal-read contract are separately qualified.

## Next PRIMARY frontier

Capture L7 tail positions 1536--1539 at pre-write K/V and attention-read
boundaries, including actual per-query causal length. Then rerun P640/P1664
state and continuation checks. Do not reopen the serializer mapping or add
another geometry experiment.

No new math kernel or unrelated performance optimization was implemented.
