# EXP-0467 — GQA6 `v_dot2` with FP32 Q and Q8 K assessment

**Status: REJECTED BEFORE IMPLEMENTATION** — the closest real-session test of
this arithmetic family was slower on both MI50s and failed the numerical
gate. The proposed single-cast form removes the prior residual correction,
which can only worsen its query approximation; no evidence supports spending
another GPU session on it.

## Objective and hypothesis

Test whether the direct-global Q8-K Stage1 path can replace its eight scalar
FP32-Q × Q8-K products per lane with four `v_dot2_f32_f16` operations. The
candidate would cast Q and dequantized Q8 K fragments to FP16 inline, keep
FP32 accumulation, preserve the ordered per-key softmax/PV recurrence, and
leave Q8 V unchanged.

This targets a measured hot stage: EXP-0436 attributes most restored long
suffix time to GQA6 Stage1 attention; EXP-0466 measured the current direct-
global 73,099 + 1,024-token restored suffix at 20.869 s. This is a proposed
arithmetic substitution only; it does not change the attention dataflow.

## Source and reference inspection

- Control source: base `c7806fcca074f04fc51eac1ba2263d534e318be4`.
- Control implementation: `gfx906/kernels/qwen3_primitives.hip`,
  `qwen35_splitk_suffix_attn_stage1_quant_kernel`, direct-global Q8 branch.
  Each lane loads two FP32 Q vectors and a Q8 K fragment, forms eight
  FP32-by-integer products, reduces the dot, then applies the per-key scale.
- Prior same-family implementation: EXP-0430 used `v_dot2_f32_f16` with
  FP16 high and residual query terms against the Q8/Q8 cache. Its temporary
  source patch SHA-256 was
  `a351431a13326fdc1cca00cbb7f716cfd26d7b54d23723a76e95cb4088cafcf0`.
- Prior measured binary SHA-256:
  `437556a4650b13835d62a8bfc98cc45fa4cdc36c553e68240aafaba061f1f2e4`.
  That artifact was built with ROCm 7.14; this assessment does not claim it is
  the binary for the current pinned ROCm 7.2.1 environment.

## Existing full-model evidence

EXP-0430 compared the packed-Q8 GQA6 control and dot2 candidate on a restored
73,099-token Q8/Q8 session, a fixed 1,024-token suffix, and one HIP Graph
replay. GPU2 (`0000:88:00.0`) control took 22.093 s and dot2 took 23.572 s
(candidate/control 1.067x latency). GPU0 showed the same direction: 20.276 s
versus 21.584 s. Both routes completed all 16 GQA layers.

The candidate produced different logits and recurrent state on both cards;
GPU2 maximum absolute logit difference was 1.232464790 and cosine was
0.999299426. The first two greedy tokens happened to match, which did not
clear the numerical gate. This was one pair per card, enough to reject on
correctness; it is not a timing distribution.

EXP-0466 subsequently confirmed a same-source-family direct-global control
latency of 20.869 s on GPU2 for the same restored prefix/suffix shape.

## Cast-cost analysis

The candidate needs Q FP32-to-FP16 conversion and Q8-K conversion plus scale
application before the dot. `v_dot2` halves the number of pairwise dot
instructions relative to scalar products, but introduces conversions and
retains the same per-key load, reduction, softmax, and V work. The previous
high-plus-residual implementation paid for additional dot work yet still ran
6.7% slower on GPU2. Removing its residual term could reduce work, but would
also remove the correction that partially recovers FP32 query precision.
No compiler ISA/resource evidence exists for this exact single-cast variant,
so cast overhead is not claimed as measured. The real-session slowdown and
numerical failure are the deciding evidence.

## Gate decision

| Gate | Evidence | Decision |
| --- | --- | --- |
| Correctness | Prior high-plus-residual version already diverged; single-cast query is less precise by construction | Reject |
| Performance | Prior version was 6.7% slower on GPU2 and 6.4% slower on GPU0 | Reject |
| Compiler/resources | No candidate object was built; no candidate ISA, resource metadata, or binary SHA exists | Not run |
| GPU2 candidate run | Not run; existing same-family real-session result failed both correctness and performance | Not run |
| Product task | No agent/task outcome was run | Not run |

**Disposition: REJECT.** Do not add a selector or source path, build this
single-cast variant, or repeat its GPU timing. Reopen only if a materially
new representation avoids conversion cost while restoring exact-enough QK
values, or a profile proves the scalar QK arithmetic itself—not memory or
the remaining attention schedule—is the limiting operation.

## Reproducibility and scope

- Base/source SHA inspected: `c7806fcca074f04fc51eac1ba2263d534e318be4`.
- New candidate source/object/binary SHA: none; no code or binary was created.
- GPU route trace, exact logits/state, token diagnostics, and timings above
  refer to EXP-0430, not to a new run.
- No GPU was contacted for this assessment; no process, service, or device
  state was changed.
