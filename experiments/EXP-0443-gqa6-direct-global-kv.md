# EXP-0443 — GQA6 direct-global Q8 K/V consumption

**Status:** IN PROGRESS — GPU result pending
**Control:** packed-U128 production-arithmetic GQA6 at `1e5b473`
**Candidate branch:** `perf/m31-gqa6-cross-token-gpu0`

## Objective and bottleneck

Reduce the restored long-context Stage1 attention time. EXP-0412 measured
Stage1 at 25.759 s of a 30.165 s 1,024-token suffix tile. EXP-0436 measured a
98.86% L2 hit ratio, no LDS bank conflicts, and high aggregate LDS-wait issue
counts (not a wall-time stall percentage).

## Prior-art gate

Cross-token cooperative CTA tiling is closed: EXP-0413/0427 rejected larger
tiles, EXP-0425 retained only a roughly 1% exact paired-token gain, and
EXP-0435 found negligible Split-K boundary overhead. This candidate does not
change the query tile. A repository search found no prior direct-global GQA6
route. EXP-0415's streamed-U4 candidate still used LDS staging and is not this
design.

## Hypothesis and change

The current three-wave GQA6 CTA stages each four-row packed Q8 K/V tile through
LDS, synchronizes, and then each query wave reads its fragment. With a hot L2,
having each query wave load its own packed K/V directly may remove LDS writes,
reads and barriers even though it duplicates global load instructions.

Only the K/V operand route changes. Preserve the six query-head mapping,
per-query Split-K interval, QK products, reduction order, scale application,
online-softmax updates and output accumulation. The route is opt-in and
included in HIP Graph cache identity.

## Gates

- **Compiler:** gfx906 code object, VGPR/SGPR, spills, scratch and LDS. Reject
  spills or unexpected private state.
- **Correctness:** one restored 73,099-token session plus a tokenizer-backed
  1,024-token graph suffix. Require exact full logits and recurrent-state
  trace, equal checked tokens, finite output and correct final position.
- **Performance:** one matched session pair against packed-U128 GQA6. Keep
  only for exactness and at least 10% suffix improvement. Otherwise reject.
- **Safety:** GPU0 only, BDF `0000:04:00.0`, telemetry active, warning at 80 C,
  stop at 85 C.

## Result

Pending.
