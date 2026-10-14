# EXP-0410: Native-Q8 GQA6 tiled attention

## Question

Can the production Q8 attention math load each KV-head's Q8 K/V tile once and
reuse it across all six GQA query heads, while preserving production QK score
arithmetic and full-model logits?

## Candidate

The opt-in `MIINFER_M31_Q8_GQA6_NATIVE_TILED=1` path assigns one CTA to one KV
head and four query tokens. The CTA covers the 24 query rows for the six query
heads associated with that KV head (32 row slots, with the final eight masked).
It stages native Q8 keys and values, then reuses them across all six heads.
The query stays FP32. For every Q/K dimension it computes
`key = float(q8_byte) * key_scale`, followed by
`dot = fmaf(q_fp32, key, dot)`, then applies the production attention scale.
There is no Q quantization, residual plane, or integer-dot approximation.

This is an isolated opt-in kernel candidate. Production dispatch remains the
default unless explicitly enabled. The suffix graph cache key includes the
selector so a graph captured with another attention path is not reused.

## Offline checks

- Base: `3805063`.
- Build: `docker.io/mixa3607/rocm-gfx906:7.14-complete`, local image digest
  `sha256:622c025824d465faa0f5e59eea4f723845032011df273c3b11e25031965fc428`.
- Configure and target `miinfer-m31-fp16-full-model`: PASS for gfx906.
- HIP compiler: Clang 23.0.0; build target compiled and linked.
- Candidate kernel resources: 55 SGPR, 85 VGPR, zero VGPR spill, zero scratch,
  two waves per SIMD. The compiler reports zero statically allocated LDS;
  launch-time dynamic shared memory is supplied by the wrapper.
- `git diff --check`: PASS.
- No GPU run has been performed yet. The local build image is not the pinned
  runtime image and is used only for the compile/resource gate.

## GPU gate

Next: build this commit in the pinned Machinist image and run one tokenizer-
backed 2K production-Q8 versus candidate pair on MI50 GPU0, PCI BDF
`0000:04:00.0`, unique ID `0x21678e17348c2f7`. Before launch, verify the card
from inside the process and check the host reports it idle. Keep the existing
thermal guard. Stop on non-finite logits, cosine below `0.9999`, top-1 or
second-token mismatch, or the thermal stop threshold. Report latency only if
the correctness gate passes. A 64K run requires the same-GPU production
control and candidate with identical build, model, prompt, and capacity.

## Provenance

Prior source review examined the public gfx906 llama.cpp fork at
`1373e889...`, MX-Llama at `71a0624...`, and Strata at `a0116b2...`. Only the
general dataflow idea of sharing KV loads across GQA heads informed this
candidate. No source code was copied from those projects; this implementation
is written against MIInfer's existing kernel and cache interfaces. No new
third-party license terms or source files are introduced.
