# EXP-0443 — GQA6 direct-global Q8 K/V consumption

**Status:** SCREEN COMPLETE — exact and repeatably faster, but below the
predeclared 10% retain threshold
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
- **Performance:** matched session pairs against packed-U128 GQA6. The
  predeclared retain threshold was at least 10% suffix improvement.
- **Safety:** GPU0 only, BDF `0000:04:00.0`, telemetry active, warning at 80 C,
  stop at 85 C.

## Result

The candidate code is committed on this branch as
`8b90661ea8ca0d1930749b761f0085efba5067f7` and was rebuilt with pinned ROCm 7.2.1 image
`localhost/miinfer-m31-rocm721-libdw:20261014T094012Z`. The gfx906 code object
metadata for direct-global GQA6 shows 60 VGPR, 50 SGPR, zero VGPR/SGPR spills,
and zero private bytes; packed-U128 control uses 71 VGPR and 46 SGPR, also with
no spills/private bytes. Disassembly of the candidate function contains 16
direct global load instructions and no `ds_read`, `ds_write`, or `s_barrier`.
Executable SHA-256:
`4d28196fa670a5cf5d193aab9e5a3ece639c4f9b4d8b633d75c3e7a5188e23ba`.

Three paired samples ran serially on MI50 GPU0, BDF `0000:04:00.0`, with the
same Qwen3.8-27B Q4_K_M model, 73,099-token restored Q8/Q8 session, and the
first 1,024 tokens of the tokenizer-backed 663,758-byte prompt. Each path
executed one HIP Graph suffix tile and reached position 74,123.

| Pair | Packed-U128 control | Direct-global candidate | Exact logits/state/tokens | First / second token |
| --- | ---: | ---: | --- | --- |
| 1 | 20.394 s | 18.806 s | yes | 18369 / 198 |
| 2 | 20.405 s | 18.795 s | yes | 18369 / 198 |
| 3 | 20.411 s | 18.805 s | yes | 18369 / 198 |

Across all three pairs, control median was 20.405 s and candidate median was
18.805 s, a 1.085x ratio (7.8% lower suffix time). Control range was
20.394–20.411 s; candidate range was 18.795–18.806 s. All comparisons had
exact full logits and recurrent-state traces, equal checked tokens, finite
values, and all 16 attention layers dispatched through the intended control or
candidate route. This confirms a reproducible gain, but it remains below the
predeclared 10% threshold. No cold-prefill or multi-GPU claim is made.

Peak junction temperature was 68 C. The 80 C warning / 85 C stop guard sampled
normally and did not act; both run containers and guard supervisors exited 0.

Raw run and telemetry records are on Machinist at
`/home/machinist/m31-gqa6-longctx-results/EXP-0443-direct-global-gpu0/`.
The first pair is in `run.log` and `telemetry.log`; pairs 2–3 are in
`repeat-r2-r3/run.log` and `repeat-r2-r3/telemetry.log`. The candidate remains
opt-in and isolated on this branch. It is a repeatable suffix improvement, but
it missed the original ≥10% gate and is not production-default or cold-64K
qualified.
