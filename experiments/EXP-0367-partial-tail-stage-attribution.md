# EXP-0367 — Exact partial-tail stage attribution and state oracle

## Question

Can an aligned partial-tail route be compared against the scalar fallback with
complete state and continuation evidence, and where does its first semantic
divergence occur?

## EXP-0365/0366 evidence

EXP-0365 measured the P1022 remainder pathology; EXP-0366's opt-in route
changed routing but was not qualified. EXP-0367 therefore used the clean
aligned boundary `P640 = B512 + B128` first. P1022 was deliberately not used
for attribution.

## State-oracle design

The new `--exp0367-state-output FILE` option is opt-in and test-only. It reuses
the existing M26-C state serializer and writes:

- every recurrent state and convolution/history buffer;
- active full-attention K/V up to the prompt position, respecting the stored
  representation conversion;
- a final-hidden FP32 dump at `FILE.final-hidden.f32`;
- a snapshot importable by the ordinary direct decode route.

The hermetic qualified preset now preserves only the EXP-0367 diagnostic
selector and profile selectors; ordinary runs still clear experimental
overrides. No production allocation or execution path changes unless the
explicit export option/selector is supplied.

## Oracle validation

Exact P640 was run once with the default fallback and once with
`MIINFER_EXP0366_PARTIAL_TAIL=1`. The default and candidate snapshots were
compared by layer and active cache contents. The candidate was then restored
through ordinary direct decode for eight greedy tokens.

| Contract | Result |
|---|---|
| Recurrent state through L2 | byte-identical |
| First differing object | full-attention layer 3 active K/V |
| First differing active position | 515, K element offset 64 |
| K/V maximum absolute difference | `0.00195312` |
| Final hidden | differs; snapshots have distinct SHA-256 fingerprints |
| 8-token continuation | identical IDs: `561,3841,13477,37550,33075,888,279,15217` |
| Base positions 1536/448 and later | not run; stop at first divergence |

The first divergence is therefore localized to the L3 full-attention partial
KV-write/causal contract at the first partial-tail region, not to recurrent
state. Continuation parity is useful evidence but does not erase the recorded
active-K/V numerical mismatch.

## Aligned-tail timing and route evidence

| Case | Prefill wall | Total wall | Allocation |
|---|---:|---:|---:|
| P640 default fallback | 16,028.51 ms | 113,744.57 ms | 21,993,243,028 bytes |
| P640 opt-in route | 3,255.38 ms | 97,003.06 ms | 21,993,243,028 bytes |

The opt-in route substantially reduces prefill wall, but these are not a
promotion claim: the state contract diverges numerically and the CLI's total
wall includes the existing zero-token terminal path. A clean stage-exclusive
GPU timing breakdown was not claimed. The diagnostic profile was not used as
an A/B result because its event synchronization changes execution.

## Execution-mode and reconciliation status

The existing aggregate profile did not provide a trustworthy exclusive
reconciliation for this run. The new counter fields distinguish partial-tail
`layer.run` calls when the profile is enabled, but a qualifying counter capture
was not completed after the preset allow-list fix. Therefore remaining
per-token calls and >=95% wall reconciliation are recorded as **not yet
measured**, not zero or accounted.

## Amdahl ceiling

The inherited P1022 pathological excess is approximately `78.48 s`. P640
shows that the opt-in route can remove substantial partial-tail prefill wall,
but EXP-0367 does not assign a P1022 share to a single stage. No stage-level
Amdahl claim is made beyond the localized K/V correctness divergence.

## Decision

**LEARN** — the state oracle is now usable for arbitrary prefill snapshots,
and the first correctness divergence is localized to L3 active K/V at the
partial-tail boundary. The aligned route is not promoted.

## Next PRIMARY frontier

Qualify the exact L3 full-attention partial K/V-write/causal contract: compare
the relevant preparation, RoPE, KV-write, and attention inputs at the first
divergent position using the same oracle. Do not tune tail geometry or claim a
performance promotion until that numerical contract is accepted or repaired.

No production optimization or new math kernel was implemented.
