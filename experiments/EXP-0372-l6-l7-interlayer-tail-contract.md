# EXP-0372 — Isolate L6 → L7 inter-layer B128 tail contract

## Question

Does the P1664 B128 candidate first diverge in L6 input/state, L6 output, or
the L6 → L7 handoff?

## Method

The exact P1664 prompt and `m25_hi_qualified` environment were reused. Control
and candidate snapshots were exported at the same position. The candidate
enabled only the existing `MIINFER_EXP0366_PARTIAL_TAIL=1` route. An opt-in,
test-only capture recorded four L6 rows at positions 1536--1539 and the
corresponding L6 recurrent state/history.

## Result

L6 input already differs at the first captured boundary:

| Boundary | Max abs | RMS | Mean abs |
|---|---:|---:|---:|
| L6 input, positions 1536--1539 | `0.0280781` | `0.00125152` | `0.000955297` |

The first divergence is upstream of L6 execution. L6 output and L7 input were
not treated as origin evidence after this stop condition. The available output
captures are finite but do not establish a handoff defect.

## State and history

| Layer | State max abs | State RMS | History max abs | History RMS |
|---:|---:|---:|---:|---:|
| L5 | `0.000877306` | `1.04101e-05` | `0.0194431` | `0.00382054` |
| L6 | `0.00110671` | `1.22648e-05` | `0.0360839` | `0.00541371` |

This supports an upstream recurrent-state/history contract issue rather than
an L6 final-add, fused normalization, pointer, or L6 → L7 handoff defect.

## Decision

**LEARN** — do not add an L6 handoff or attention-tail fix. Reopen at L5 → L6
only after refreshed full-model attribution identifies that boundary as
material. No timing, resource qualification, or performance promotion was
attempted after the first semantic divergence.

## Next PRIMARY

Refresh full-model prefill attribution and then, if still material, qualify the
L5 → L6 input/state contract at P1664. Do not start EXP-0373 or implement a
new performance kernel from this record alone.

No new performance optimization was implemented.
