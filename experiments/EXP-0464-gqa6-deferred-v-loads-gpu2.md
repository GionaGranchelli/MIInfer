# EXP-0464 — GQA6 KQ-first deferred V loads, GPU2

## Objective and baseline

Test one materially different Stage1 operand lifetime on GPU2 for the M31
Qwen3.8-27B Q8-KV restored suffix. Keep the model, cache, persistent state,
32 splits, `(64,3)` CTA, four-key microtile, FP32 arithmetic/update order,
Stage2 output/workspace contract, and HIP Graph route semantics fixed.

The direct-global GPU2 control is about 20.98 s for the restored 73,099-token
prefix plus 1,024-token suffix. It dispatches all 16 attention layers and
matches outputs/state. EXP-0412 attributed 85.5% of a comparable suffix to
Stage1. EXP-0463 is diagnostic only: direct-global had ~99.63% L2 hits and
~172.9k VALU / ~179.9k active-VALU instructions per wave; packed-U128 had
~98.85% hits and ~166.2k / ~173.7k. This does not establish a wall-time limit
or imply packed staging is faster.

## Reference and candidate

The local gfx906 reference is `ggml/src/ggml-cuda/fattn-tile.cuh` at the
`/home/gionag/Development/llama.cpp` checkout (`91c631b21`). Its tile computes
KQ before loading V tiles into reused shared workspace. Apply only that
operand-lifetime ordering to direct-global Q8 GQA6: calculate the existing
four scores first, then load each Q8 V row and scale immediately before its
existing ordered online-softmax/PV update. No score, reduction, scale, max,
sum, or accumulator arithmetic changes.

This is not a repeat of EXP-0457's query/key tile: it adds no query tiling,
LDS, cross-head sharing, or row reuse. It tests whether shortening the Q8 V
register live range reduces VGPR pressure enough to improve the current
L2-hot, VALU-heavy schedule.

## Gates

- **Resource:** build the opt-in exact gfx906 code object first. Record VGPR,
  SGPR, spills, scratch/private bytes, LDS, workgroup, and wave size. Reject
  before GPU use on spills, unexpected scratch/LDS, or a resource regression
  without a clear occupancy benefit.
- **Correctness:** one matched GPU2 restored-session pair, 73,099 + 1,024,
  with route trace in all 16 attention layers, position 74,123, nonempty
  recurrent state, finite logits/state, max-abs <= 0.01 and cosine >= 0.9999
  for logits and state, matching checked tokens, and no lost state/truncation.
  Exact equality is diagnostic.
- **Performance:** one same-binary direct-global control/candidate kill
  screen; require at least 10% lower suffix wall time before a cold 64K run.
- **Safety:** GPU2 BDF `0000:88:00.0` only; warn at 80 C and stop at 85 C.
  GPU0 remains excluded. A real coding-task continuation is required before
  any cold 64K run if numerical outputs are not exact.

## Result

Pending compiler/resource gate.
