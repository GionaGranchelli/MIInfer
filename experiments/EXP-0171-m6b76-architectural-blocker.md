# EXP-0171 — M6-B76 architectural blocker report

## Gate

Three materially different structural families have now failed to produce the
required improvement:

| Family | Evidence | Result |
|---|---|---:|
| persistent decoded Q4_K metadata | EXP-0140 | `+0.03%/-0.17%` |
| expanded Q4_K Gate/Up representation | EXP-0158 | correctness failure |
| Gate+Up+SwiGLU fused execution | EXP-0170, including reference-mapped two-wave revision | `-8.37%/-7.8%` |

The escalation rule is met. Do not generate another Bxx geometry or fusion
variant.

## Current differential

| Item | MIInfer | pinned llama.cpp | Implication |
|---|---:|---:|---|
| whole-token time | `71.913 ms` GPU event | approximately `44.8–45.1 ms` | `27 ms` gap |
| layer time | `68.627 ms` | not exposed | layers are the target |
| dispatches/token | `1,553` | not runtime-counted | launch count is not sufficient evidence |
| estimated model-weight stream | approximately `17.1 GB` | approximately `17.1 GB` | required weight work is comparable |
| estimated effective weight rate | `~238 GB/s` (`17.1/0.0719`) | `~380 GB/s` (`17.1/0.045`) | MIInfer leaves substantial HBM/compute throughput unused |
| FFN Gate/Up | separate MIInfer MMVQs | fused Q4_K MMVQ+GLU exists in source | fusion alone does not transfer without the reference inner loop |
| Q4_K metadata | packed, decoded/staged in kernels | packed Q4_K MMVQ decode | metadata relocation is not the gap |
| intermediate traffic | explicit Gate, Up, SwiGLU boundaries | fused Gate/Up/GLU boundary | measurable, but insufficient in tested kernel |

The weight-rate figures are model-stream estimates, not hardware-counter
measurements. The model is mostly weight-streaming at batch 1, and MI50's
approximately 1 TB/s peak makes both values evidence of implementation
underutilization rather than a physical bandwidth ceiling.

## Conclusion

The remaining gap is principally **quantized matvec throughput and instruction
efficiency**, with secondary intermediate-memory traffic. It is not primarily
CPU orchestration, allocation, LM head, launch count, or Q4 metadata location.

The pinned source's important advantage is a fully tuned architecture-specific
MMVQ inner loop: it packs Q4_K words into the exact `v[2]`/`u[]` arrangement,
decodes scale/min metadata into compact local values, uses `VDR_Q4_K_Q8_1_MMVQ
= 2`, and performs fused postprocessing. MIInfer's equivalent path has the
same broad decomposition but pays a different Q4_K decode/access cost. The
failed fusion prototypes confirm that duplicating this path in one kernel
increases register pressure instead of exposing the reference's throughput.

## Required redesign

The next campaign must be a kernel-laboratory port of the reference's exact
Q4_K×Q8_1 `vec_dot` representation and GCN launch contract, measured first as
an isolated exact-shape primitive. It must compare instruction/register use,
transactions, and effective bandwidth against the current primitive before
being connected to the model. If that direct primitive does not approach the
reference rate, MIInfer needs a new gfx906-native packed weight format and
model-load transformation; more executor-level fusion cannot close this gap.

No production implementation is selected by this report.

## Re-evaluation — EXP-0172

The required reference-inner-loop campaign was run as an isolated schedule
port: Q8 words and half scales were preloaded into local values before the
unrolled DP4A loop. Native replay passed and decode allocations remained zero,
but the five-run TG64 median was 14.269 tok/s versus the 14.280 tok/s control
(-0.08%). The variant was reverted and recorded as **REJECT** in EXP-0172.

This rules out simple instruction ordering/load hoisting as the missing
mechanism. The remaining redesign is a new gfx906-native weight layout with a
measurably different access pattern, or a larger execution-plan rewrite that
shares work across incompatible projections. Both require a new kernel-lab
campaign and cannot be justified by another existing-kernel variant.
