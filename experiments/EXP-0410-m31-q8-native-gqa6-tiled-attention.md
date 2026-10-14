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
- Initial pinned-image configure failed because CMake did not search the
  image's `/opt/rocm-7.2.1` prefix. Re-running with
  `CMAKE_PREFIX_PATH=/opt/rocm-7.2.1` fixed discovery; no source or build-image
  changes were needed.

## GPU gate

For every full-model gate, stop on non-finite logits, cosine below `0.9999`,
top-1 or second-token mismatch, or the thermal stop threshold. Report latency
only if the correctness gate passes. A 64K run requires the same-GPU production
control and candidate with identical build, model, prompt, and capacity.

## Pinned 2K full-model gate

- Machinist GPU0 was idle before launch: BDF `0000:04:00.0`, gfx906, unique ID
  `0x21678e17348c2f7`. The process itself printed
  `device_bdf=0000:04:00.0 architecture=gfx906:sramecc+:xnack-`.
- Image digest:
  `localhost/miinfer-m31-rocm721-libdw@sha256:422b5d15d384d78cab724a0190ef739d31212674ebc1ae8f0f6cf538b1f5ea8f`;
  local image ID `f5f33d8e8efbe77f0a08335140f9c9a162202067189499be9dbddd2eb214afa4`.
  Runtime HIP reports ROCm 7.2.53211; pinned image uses Clang 22.0.0.
- Model SHA-256:
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.
- Binary SHA-256:
  `7453c42f11a4773ab427f8e74cae8e2505a45f2d9b558722bcf5abb05761bb82`.
- The pair used 2,048 deterministic token IDs from the benchmark's existing
  sequence and capacity 65,536. Q8/Q8 control ran first, then the candidate.
- Correctness passed: finite logits, max absolute difference `0.181160450`,
  cosine `0.999952746`, and matching first/second token IDs (`14` / `16`).
- Prefill: production control `9.344 s`; candidate `8.949 s`; one-pair ratio
  `1.044x` (4.4% faster). This is a single short-prompt pair, so it is not a
  qualified speed claim. The difference is modest and order-sensitive; a
  long-context paired run is needed to tell whether GQA load reuse matters at
  the target context length.
- Thermal wrapper sampled junction temperature 99 times, max `58 C`; no 80 C
  warning, 85 C stop, telemetry failure, or timeout. GPU0 returned to 0% use
  and about 10.9 MB VRAM after the run.

## Provenance

Prior source review examined the public gfx906 llama.cpp fork at
`1373e889...`, MX-Llama at `71a0624...`, and Strata at `a0116b2...`. Only the
general dataflow idea of sharing KV loads across GQA heads informed this
candidate. No source code was copied from those projects; this implementation
is written against MIInfer's existing kernel and cache interfaces. No new
third-party license terms or source files are introduced.
