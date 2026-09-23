# EXP-0370 — Qualify L7 partial-tail K/V and causal contract

## Question

At L7 tail positions 1536--1539, does candidate/control divergence begin in
K/V production, KV storage, or causal attention reads?

## EXP-0368/0369 evidence

EXP-0368 classified P640 K/V drift as expected batched numerical behavior.
EXP-0369 proved that P1664 changes route only at the real tail
`base_position=1536, count=128`; the prior position-768 report was a state
serializer indexing artifact. Same-route pairs were byte-identical, and the
L7 cache prefix `[0,1536)` was unchanged.

## K/V pre-write captures

An opt-in L7 capture recorded projected K and the exact V buffer supplied to
the KV store for the first four tail positions. Candidate and control were
compared at the same semantic positions.

| Position | Projected K max / mean / RMS | Write-input V max / mean / RMS |
|---:|---:|---:|
| 1536 | `3.08060 / 0.404974 / 0.616314` | `1.69332 / 0.257146 / 0.364434` |
| 1537 | `3.61697 / 0.538873 / 0.754392` | `3.36653 / 0.382992 / 0.525894` |
| 1538 | `2.82921 / 0.559123 / 0.742816` | `3.02697 / 0.427013 / 0.576215` |
| 1539 | `8.65410 / 0.498809 / 0.810701` | `3.07472 / 0.357677 / 0.527725` |

The first captured boundary already differs for both K and V. Therefore the
KV writer is not the origin of this divergence. K's post-normalization and
post-RoPE values were not separately captured; the next upstream boundary is
the primary follow-up.

## Stored-cache comparison

The existing state snapshots were split by K/V and compared at L7:

| Position | Stored K max / mean / RMS | Stored V max / mean / RMS |
|---:|---:|---:|
| 1536 | `0.02084 / 0.006663 / 0.008158` | `0.03613 / 0.008238 / 0.010274` |
| 1537 | `0.02832 / 0.007999 / 0.010109` | `0.03760 / 0.007980 / 0.010383` |
| 1538 | `0.03320 / 0.010619 / 0.013123` | `0.04150 / 0.010937 / 0.013918` |
| 1539 | `0.02393 / 0.006987 / 0.008401` | `0.03738 / 0.009830 / 0.012229` |

No additional cache-write amplification was observed relative to the already
differing inputs. The prefix `[0,1536)` remained identical.

## Actual causal-read bounds

The opt-in device trace ran in the FP16 batch attention kernel and reported:

| Query | Base | Local token | Cache length | First readable | Last readable |
|---:|---:|---:|---:|---:|---:|
| 1536 | 1536 | 0 | 1537 | 0 | 1536 |
| 1537 | 1536 | 1 | 1538 | 0 | 1537 |
| 1538 | 1536 | 2 | 1539 | 0 | 1538 |
| 1539 | 1536 | 3 | 1540 | 0 | 1539 |

The kernel physically has the complete tail in cache before launch, but its
runtime loop stops at the per-query causal length. No query read future KV.

## Attention output and continuation

The first unique divergence was already established at projected K/V, so the
experiment stopped before broadening capture to attention output, O, or FFN.
No new P640 or P1664 continuation was run in EXP-0370; the qualified P640
continuation from EXP-0367 remains exact for eight tokens. This stop follows
the protocol's first-divergent-stage rule.

## Decision

**LEARN** — the first observed candidate/control difference at L7 tail
positions 1536--1539 is present in projected K and V store inputs, before KV
storage. The causal-read contract is correct for the four inspected queries;
there is no evidence of a future read or prefix overwrite. No local fix was
made because the exact upstream projection/preparation expression has not yet
been isolated.

## Next PRIMARY frontier

Qualify the exact L7 B128 projection/preparation contract, separating K
normalization/RoPE from V production and comparing it with an accepted B512
stage envelope. Do not modify the KV writer, causal bounds, or performance
geometry until that upstream boundary is understood.

No new math kernel or unrelated performance optimization was implemented.
