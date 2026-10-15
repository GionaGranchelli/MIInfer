# EXP-0455 — Four-wave GQA6 query/head tile

**Status:** IN PROGRESS — compiler/resource gate pending

## Objective

Test one source-shaped gfx906 mapping for the active GQA6 Q8/Q8 attention
frontier. An CTA handles four adjacent queries and two adjacent query heads
from one KV group (eight independent query rows), using four Wave64s. One Q8
K/V tile is staged once in LDS and consumed by all eight rows.

This changes the workgroup geometry and head grouping relative to the rejected
EXP-0427 tile4 (384 threads / six waves) while keeping its exact cross-query
reuse premise. This is a bounded re-evaluation of that geometry, not a tile
size sweep. Prior evidence remains adverse: EXP-0413/0427 rejected cooperative
query tiling, EXP-0435 found split-boundary overhead negligible, and EXP-0443
found the current direct-global one-token GQA6 route faster than LDS staging.

## Hypothesis and one variable

Four Wave64s may keep a tile4 CTA at the natural 256-thread gfx906 width and
reduce per-CTA query/head state enough to improve occupancy and amortize one
barrier across the eight query rows. The sole variable is this Stage1 row
mapping; Q8 arithmetic, per-query Split-K intervals, key order, online softmax,
stage2, model/session, graph and suffix remain fixed.

The candidate stages the union of the four query intervals once, masks each
row to its own original interval, and consumes valid keys in increasing order.
The direct-global GQA6 path remains the control.

## Gates

- **Resource:** compile for gfx906 in the pinned ROCm image; require an actual
  256-thread / four-wave launch, zero register spills, zero private bytes, and
  no extra barrier loop beyond the existing one-barrier-per-four-key tile.
  Reject if resource use/codegen is no better than the rejected six-wave tile.
- **Correctness and route:** one restored 73,099-token Q8 session plus the
  tokenized 1,024-token suffix. Require graph route trace on all 16 layers,
  exact full logits and recurrent state, equal checked tokens, finite values,
  and position 74,123.
- **Performance:** one direct-global control/candidate pair on GPU2
  `0000:88:00.0`; continue only with exactness and at least 10% lower suffix
  latency. No repeat or cold-64K run unless the first pair passes.
- **Safety:** use the verified pinned image/model/session, confirm GPU2 BDF and
  idle KFD workloads, retain the 80 C warning / 85 C hard stop, and do not use
  GPU0.

## Expected impact and kill rule

The tested object is the measured dominant attention Stage1 path. Stop if the
compiler recreates a six-wave/high-pressure schedule, adds spills/private
state, dispatch is not selected on all layers, any correctness check fails, or
the suffix improvement is below 10%. If rejected, keep the direct-global
production arithmetic path and record the geometry result; do not try another
tile or wave count.
