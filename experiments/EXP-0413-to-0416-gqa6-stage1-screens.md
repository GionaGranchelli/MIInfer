# EXP-0413 to EXP-0416 — GQA6 Stage1 optimization screens

**Status:** REJECT all four candidates
**Control:** bit-identical production-arithmetic GQA6
**Workload:** restored 73,099-token Q8 session, 1,024-token suffix, GPU2
`0000:88:00.0`, actual HIP Graph replay

These were one-sample full-model screens. All guards passed (maximum observed
temperature below 80 C). Exactness means full logits, recurrent-state trace and
checked greedy tokens matched the GQA6 control.

| Experiment | Candidate | GQA6 time | Candidate time | Exact? | Decision |
| --- | --- | ---: | ---: | --- | --- |
| 0413 | Cross-token K/V reuse, six-wave CTA | 30.164 s | 35.930 s | Yes | REJECT, 19% slower |
| 0413 | Cross-token K/V reuse, three-wave CTA | 30.170 s | 45.676 s | Yes | REJECT, 51% slower |
| 0414 | U8 staging / online-softmax unroll | 30.150 s | 54.188 s | Yes | REJECT, 80% slower |
| 0415 | Streamed U4 inner loop | 30.128 s | 37.733 s | Yes | REJECT, 25% slower |
| 0416 | Dot4 QK with three query residual planes | 30.160 s | 19.277 s | No | REJECT; max logit error 0.717 |

The U8 object reports 58 VGPR spills versus 2 for GQA6. The streamed kernel
reduced VGPR use to 51 with no spills but remained slower, so nominal VGPR
count alone did not explain Stage1 runtime. Dot4 had a large isolated/full
suffix speedup but missed the predeclared `max_abs <= 0.01` gate and its
recurrent state diverged; matching generated tokens did not make it acceptable.

## Architectural learning

The two exact cross-token schedules failed to convert K/V reuse into lower
wall time. U8 processing and serial streamed consumption also lost. Do not
repeat these shapes as parameter sweeps. The remaining bounded exact-arithmetic
screen is to reduce the instruction count of the existing GQA6 U4 LDS staging
without changing bytes, QK/V arithmetic, softmax order, or split boundaries.

Raw result directories are under
`/home/machinist/m31-gqa6-longctx-results/EXP-0413-cross-token/`,
`EXP-0414-gqa6-unroll8/`, `EXP-0415-gqa6-streamed/` and
`EXP-0416-dot4-residual3/`.
