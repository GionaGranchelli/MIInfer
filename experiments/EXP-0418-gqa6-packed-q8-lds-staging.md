# EXP-0418 — Packed Q8 staging for production-arithmetic GQA6

**Status:** SCREEN PASSED — exact and 1.36x faster; repeated timing gate pending
**Control:** production-arithmetic GQA6, restored 73,099-token suffix ~30.2 s

## Objective and bottleneck

Reduce GQA6 Stage1 instruction overhead on the real restored-session workload.
The corrected profile attributes 25.759 s of a 30.165 s tile to Stage1, but
does not isolate the fraction spent staging K/V into LDS. The maximum possible
whole-tile benefit is therefore unknown and is bounded by the measured Stage1
time.

## Hypothesis and independent variable

GQA6 currently stages the same 1,024-byte K tile and 1,024-byte V tile into
LDS using one scalar int8 load/store per byte. For the fixed `head_dim=256`,
U4 tile, each row begins at a 4-byte-aligned offset. Copying four adjacent
bytes through one aligned 32-bit load/store may reduce staging instruction
count while preserving the shared byte layout and every arithmetic operation.

Only the K/V LDS copy width changes. Keep 4-key online-softmax order,
production QK/V arithmetic, scales, causal mask, per-query Split-K boundaries,
graph, workspace, model and suffix fixed. The caller retains a scalar fallback;
the packed route is opt-in.

## Reference, resource and correctness gates

- The reference is the existing production-arithmetic GQA6 implementation;
  no external implementation has a directly comparable stateful MI50 path for
  this kernel. This is a localized transformation of that proven route.
- Inspect gfx906 ISA to verify packed global loads/shared stores were emitted;
  collect VGPR/SGPR, spills/private bytes, and dynamic LDS before timing. Stop
  if spills appear or the compiler already emits an equivalent copy schedule.
- On the restored 73,099-token session, require HIP Graph dispatch proof,
  finite logits, bit-exact full logits and recurrent-state trace, unchanged
  first/second greedy tokens, and correct final position.
- Screen one matched 1,024-token suffix on GPU2. Keep only if exact and at
  least 10% faster than the ~30.2 s GQA6 control. Otherwise reject; do not tune
  launch geometry under this experiment.
- Maintain 80 C warning / 85 C hard stop. Stop on HIP error or wrong route.

## Kill and next action

If packed loads/stores are not emitted, revert before GPU timing. If the exact
route is less than 10% faster or fails any correctness/resource gate, record
the result and move to a materially different attention schedule. Do not use
this result to claim cold-64K improvement; a full cold run follows only after a
material real-model suffix win and a refreshed end-to-end profile.

## Result

The selected implementation uses `uint2` K/V copies, with K stored before V
to limit live vector state. Correct CTA launch bounds are now declared as 192
threads for GQA6; the same bound is used by control and candidate. This was
needed because the compiler otherwise recorded a 1,024-thread maximum for the
actual 192-thread launch and generated extra spills.

In the pinned image, the candidate reports 71 VGPR, 46 SGPR, zero VGPR/SGPR
spills and zero private-segment bytes; the same-build GQA6 control reports 68
VGPR, 49 SGPR and zero spills/private bytes. Disassembly contains
`global_load_dwordx2` and `ds_write_b64` for packed staging. Scalar byte loads
and stores remain only in the unchanged tail path. Dynamic LDS is unchanged.

The guarded 1,024-token graph-replay pair on the restored 73,099-token session
used GPU2/BDF `0000:88:00.0` and the pinned ROCm 7.2.1 image:

| Route | Restore | Suffix |
| --- | ---: | ---: |
| GQA6 control | 2.456 s | 30.520 s |
| Packed GQA6 | 2.444 s | 22.434 s |

This is 1.360x faster in one screening pair. Full logits and recurrent-state
trace were bit-exact; both checked greedy tokens matched; the session position
reached 74,123; the graph route trace confirmed packed dispatch with dynamic
state in all 16 attention layers. The guard passed; junction temperature
peaked at 57 C. No cold-prefill claim is made. Repeat the matched pair before
promoting the opt-in selector beyond this experiment.
