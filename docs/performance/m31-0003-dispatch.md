# M31-0003 — Host dispatch and synchronization

## Production timelines

The production `run`, `chat` and `serve` entry points instantiate
`PrefillV2Model`; `Qwen35RuntimeEngine` is not this path.

| Route | Host/device sequence | Confirmed cost / unknown |
|---|---|---|
| Cold prefill | For each ≤512-token tile: async token-ID H2D, `forward()` over all 16 blocks. Then LM-head logits D2H and stream sync for CPU sampling. | One host dispatch loop per tile; no cold-prefill graph capture. Long-context host overhead not isolated. |
| Persistent suffix | Restore GDN if not resident; process only suffix tiles. Full 512 tiles may use suffix HIP graph, residual tiles use ordinary `forward()`. Final stream sync. | Graph does not cover every suffix shape; restore and suffix timings are path-specific. |
| Ordinary decode | Replay one-token HIP graph per token; copy full logits D2H and synchronize for CPU sampling; update decode state H2D before next replay. | Correctness-oriented CPU sampling round trip is confirmed. Its long-context share is unknown. |
| Direct diagnostic decode | Launch layer path, device argmax, copy one token D2H and synchronize. | Not equivalent to `generate()` graph route; do not compare their timings as if they were. |

## Repeated boundary checkpoint copies

With `cache_prefix_after=true`, the current dirty `generate()` calls
`ReusableContext::save` after each completed 512-token boundary, overwriting
the same one-checkpoint GDN storage. Each save issues 48 state plus 48 history
D2D copies and one final-hidden copy (97 API copies). At 128K there are up to
256 boundaries: about 24,832 copy calls and 40.67 GB of cumulative checkpoint
payload copied. This is a source-derived upper estimate, not a measured time.

Agent review found that for a successful request, one final save is sufficient
to preserve the final prefix; repeated saves are not needed for final-state
correctness. However, intermediate saves may be intended as progress/cancel
behavior, and the current exception/cancellation contract is not fully tested.
This change was therefore not made in this goal. A separately reviewed
single-final-save change should test exact-prefix reuse, suffix reuse, short
prompts and failure behavior before rollout.

There is no per-layer/token VRAM allocation in the ordinary decode loop from
the inspected source. Cold prefill graphing and changing sampling location are
larger architectural changes and are not justified by existing timings.

## Decision

Host transfers and synchronization are confirmed source operations; their
latency is not yet causal evidence for the measured long-context gap. The
cheapest next evidence is existing short synchronized profiles and static
counts; GPU timing is required before changing graph, sampling or checkpoint
dispatch behavior.
