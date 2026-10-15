# EXP-0457 — Q8 FlashAttention query × key tile, GPU2

## Objective

Reduce late-context Q8 GQA6 Stage1 time on MI50 with a real multi-query ×
key-tile dataflow. Preserve Q8 KV and FP32 query/softmax/value accumulation.

## Baseline and measured bottleneck

- Frozen source control: `208a926d26aac126580bc26cd8c36ce0d2808762`.
- Direct-global Q8 GQA6 is the timing control, selected explicitly with the
  current route selector off for all experimental selectors.
- EXP-0421 attributed 17.993 s of a 22.384 s profiled restored suffix tile to
  Stage1. EXP-0443/0446 establish direct-global as the best exact arithmetic
  path, but its reported gain remains below the retain threshold.
- Test one restored 73,099-prefix + tokenizer-backed 1,024-token suffix on
  GPU2 only. GPU0 stays frozen.

## Reference and rejected work avoided

The Qwen grouped-query mapping, Q8 scales and online-softmax contract come
from the existing production GQA6 route. The dataflow uses the standard
streaming FlashAttention recurrence but keeps separate FP32 state per output
row and computes multiple rows against each staged Q8 K/V tile.

This is not the rejected 16-query fragment-reuse path (EXP-0417), the
tile-size/query-reuse screens (EXP-0425/0427), the broken four-key stable
reduction (EXP-0442/0445), the V2-0043 FP16 fragment route, or MFMA (no
gfx906 MFMA instruction was emitted in the pinned compiler screen).

## Fixed design

One CTA owns two adjacent suffix tokens, one KV head and one adjacent pair of
Q heads. Its four logical 32-lane rows are packed into two physical Wave64s;
each logical row owns one `(token, q_head)`. Each row maps lanes over D (eight
dimensions per lane) and computes four keys in ILP, then reduces each score
across lanes. Stage Q8 K/V once per CTA in row-major `[key][dimension]` LDS;
this makes QK dimension loads coalesced and lets each lane reuse its staged
dimension slice across four keys and four query rows. Compute a 4×32 score
tile with FP32 QK arithmetic;
perform a separate stable online-softmax update for each row; then reuse the
same K/V tile for each row's weighted-value update. Each row computes its own
causal length and original ceil-sized split interval. A short final query
tile masks invalid rows.

The one independent variable is the Stage1 Q×K execution/dataflow. Keep model,
Q8 cache, 32 splits, persistent state, graph replay, and output workspace
contract fixed. The feature stays opt-in and defaults off.

## Fixed-operand oracle

Before model execution, compare the implementation to a scalar FP32 reference
on a deterministic captured-operand fixture. Assert every row's token/head/KV
mapping, key scale, causal interval, split interval and partial-output offset.
Compare per-row max, sum, weighted accumulator and normalized output using
max-abs and cosine bounds; exact equality is diagnostic only because the
block-softmax association differs from the sequential production path. Include
the 73,099-prefix adjacent-token split-size transition (suffix tokens 20/21),
the 31/32 cold boundary, first/last split, empty early splits, and a partial
final BK tile. Stage the union of row intervals but mask every score/value by
that row's own original `[k_start,k_end)`.

## Gates

- **Compiler:** gfx906 object; report VGPR, SGPR, spills, scratch, LDS and
  workgroup. Stop on any spills or unexpected scratch.
- **Oracle:** all row-map assertions pass; state/output max-abs <= 1e-4 and
  cosine >= 0.99999 against scalar FP32 fixture. Any mask/head/scale/layout
  error stops the work.
- **Runtime correctness:** correct route observed in all 16 layers, finite
  full logits/state, logit max-abs <= 0.01 and cosine >= 0.9999, recurrent
  state max-abs <= 0.01 and cosine >= 0.9999, checked generated tokens
  recorded. Exact logit/state equality is diagnostic, since the tiled
  FP32-softmax association differs from the production sequence.
- **Performance:** one same-GPU2 direct-global/candidate pair on the restored
  suffix. Candidate must pass numerical gate and reduce suffix time by >=10%
  before any repeat or cold-64K run.
- **Safety:** verify BDF `0000:88:00.0`; thermal warning 80 C, hard stop 85 C.

## Decision

Pending implementation and gates. Revert code on resource, oracle, runtime
numerical, or performance failure; retain the report and raw evidence.

## Harness

The restored-session pair mode adds `MIINFER_M31_Q8_GQA6_FLASH_TILE_PAIR=1`.
It explicitly uses direct-global GQA6 for the control and toggles only the
flash-tile selector for the candidate. It emits per-layer route traces,
compares complete logits and recurrent state with the numeric bounds above,
and records exact equality and generated-token agreement as diagnostics.
