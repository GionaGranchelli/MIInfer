# EXP-0459 — GQA6 pair-head reuse schedule

**Status:** implementation pushed; pending gfx906 resource and restored-session gates

## Objective and frontier

Test one new Stage 1 schedule for the active M31 PRIMARY: materially faster,
correct long-context GQA attention for Qwen3.8-27B on one MI50, for cold
prefill and persistent suffixes. The experiment keeps the model, Q8 KV cache,
persistent state, split workspace, and HIP Graph interface fixed.

## Bottleneck and actor census

EXP-0412 attributed 85.5% of a restored late-context tile to Stage 1. The
73,099-token-prefix/1,024-token-suffix control in EXP-0458 took 20.888 s for
the suffix; its candidate did not pass the numerical or 10% performance gate.
This experiment does not infer that Stage 1 alone explains cold-prefill time.

EXP-0441's CTA census rejects split imbalance: each split receives essentially
equal key work. The current GQA6 mapping groups two query heads in one Wave64,
with one 32-lane half-wave per head. Both halves consume the same Q8 K/V tile
separately. For each key row, that is two 8-byte LDS reads per 32 dimensions
for each two-head pair. The candidate maps all 64 lanes across the 256
dimensions (four dimensions per lane), computes both query heads, and reads
each four-byte K/V fragment once per pair. It therefore halves the per-pair
LDS operand reads while keeping the same three waves per CTA and four-key LDS
tile. The dot reduction changes from two 32-lane reductions to two 64-lane
reductions; this cost may erase the load savings and is measured, not assumed.

EXP-0436 found high aggregate LDS-wait and VALU counters but no reported bank
conflicts and a 98.86% L2 hit ratio. Those counters came from packed-LDS GQA6;
they do not prove an end-to-end LDS stall fraction. The candidate directly
tests the concrete duplicated LDS reads rather than treating the counters as
wall-time attribution.

## Reference and one variable

Reference: packed-U128 production-arithmetic GQA6 from commit `8b3d553`, using
the same 32 splits, 4-key LDS tile, Q8 cache, model/session, and HIP Graph
replay. The only variable is `MIINFER_M31_Q8_GQA6_PAIR_HEAD_REUSE=1`.

The candidate changes the lane-to-dimension mapping and computes two Q heads
per Wave64. It keeps the three-wave CTA shape, K/V tile staging, per-key online
softmax sequence, split boundaries, Stage 2, state format, and graph contract.
QK reduction association changes, so exact logits are not a pass requirement;
finite output and task behavior are.

## Gates

- **Target/resource:** build against the pinned ROCm 7.2.1 image for gfx906;
  inspect the instantiated object, VGPR/SGPR, spills, private bytes, LDS,
  block dimensions, wave size, and generated K/V load and shuffle schedule.
  Stop on wrong target, spills, or unexpected LDS/resource use.
- **Correctness screen:** on GPU2 BDF `0000:88:00.0`, restore the same 73,099
  prefix, replay the same tokenizer-backed 1,024 suffix, reach position
  74,123, verify candidate route on all 16 attention layers, finite logits and
  recurrent state, token outputs, and no lost state or truncation. Record
  logit/state max-abs and cosine as diagnostics. This screen alone does not
  establish coding-task correctness.
- **Performance screen:** one matched control/candidate pair on GPU2 with the
  same binary, model, session, capacity, thermal guard, and environment except
  for the selector. Require at least 10% lower suffix wall time before any
  cold 64K run. A single pair is a kill screen, not a qualified median.
- **Thermal:** warn at 80 C and stop at 85 C. GPU0 remains the frozen control.
- **Kill:** wrong route/state/position, nonfinite values, spill or unexpected
  scratch, or less than 10% suffix improvement. Do not tune tile sizes or split
  counts after failure.
- **Success:** if all screens pass, preserve the run evidence, then move to a
  real coding-task continuation and only then a guarded cold 64K prefill.

## Prior rejection checks

This is not another split-count or macro-tile sweep (EXP-0441/0406), a
pairwise16 reduction (EXP-0432), parallel CTA K/V staging (EXP-0429), a
cross-token tile (EXP-0425/0427), or changed online-softmax association
(EXP-0458). It reassigns the 64 lanes so one Wave64 shares each loaded K/V
fragment across a query-head pair.

## Result: reject

The candidate compiled for gfx906 with 81 VGPR, 44 SGPR and no spills. On
GPU2/BDF `0000:88:00.0`, the restored session reached 74,123 tokens and the
HIP Graph trace confirmed pair-head dispatch on all 16 attention layers.
The matched 1,024-token suffix took 22.239 s on packed-U128 GQA6 and 31.537 s
on pair-head reuse: the candidate was 1.42x slower. Both outputs were finite
and the checked generated tokens matched, but final logits and recurrent
state were not exact (`logits_max_abs=0.643235207`,
`logits_cosine=0.999692678`). The screen checked only generated-token
agreement, not a coding task, so it establishes no task-correctness result.

Reject this lane remapping. Do not repeat the suffix run or proceed to cold
64K with it. Raw run and telemetry are preserved on Machinist under
`/home/machinist/m31-gqa6-longctx-results/EXP-0459-pair-head-reuse-gpu2/`.
