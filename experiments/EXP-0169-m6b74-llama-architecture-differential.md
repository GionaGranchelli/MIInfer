# EXP-0169 — M6-B74 pinned llama.cpp architecture differential

## Question

What work does pinned llama.cpp eliminate that MIInfer repeats during Qwen3.8
Q4_K_M decode?

## Evidence

MIInfer's production path in `src/qwen3_gpu_layer.cpp` launches Gate and Up
independently, then launches SwiGLU/Down-input quantization. The shared-input
optimization reuses the already-produced Q8 activation, but does not share the
weight loop, reduction, or output materialization.

Pinned `c0bc859` source:

* `ggml/src/ggml-cuda/ggml-cuda.cu` detects adjacent Gate, Up, and GLU nodes.
* `ggml/src/ggml-cuda/mmvq.cu` launches `mul_mat_vec_q_switch_fusion`.
* The fused gfx906 MMVQ kernel accumulates `tmp` and `tmp_gate` in the same
  workgroup, reads the same Q8 tile/schedule, reduces both, applies SwiGLU,
  and writes the GLU result without materializing separate Gate/Up outputs.
* Q4_K uses the GCN MMVQ table; Q5_K and Q6_K have the same Q8_1 MMVQ family,
  but Qwen3.8's FFN Gate/Up and Down weights are Q4_K.

This is verified source evidence, not an inference from operation names.

## Required differential report

| Item | MIInfer | llama.cpp | Implication |
|---|---:|---:|---|
| ms/token | 71.913 GPU event | ~44.8–45.1 | 27 ms gap |
| layer ms/token | 68.627 | not exposed | target is layer execution |
| launches/token | 1,553 (profile) | not yet runtime-counted | count is secondary; fusion changes work, not only launches |
| recurrent-layer launches | ~24 | lower with fused QKV/FFN boundaries | validate with trace |
| attention-layer launches | ~24 | lower with fused norm/RoPE/attention boundaries | validate with trace |
| estimated weight bytes/token | ~17.1 GB model stream | ~17.1 GB model stream | bandwidth is comparable at model level |
| estimated activation traffic/token | repeated Gate + Up + GLU vectors | fused Gate/Up/GLU boundary | remove intermediate FFN traffic |
| effective weight bandwidth | not measured per family | not measured per family | add kernel timestamps before final claim |
| FFN Gate/Up strategy | 2 MMVQ launches, shared Q8 input | 1 fused MMVQ launch | highest-value confirmed difference |
| FFN Down strategy | separate MMVQ | separate MMVQ | not the first target |
| recurrent QKV strategy | 3 projections | graph-level independent matmuls; no verified QKV fusion | no evidence yet |
| Q4 metadata handling | packed/decoded per selected kernel | GGUF Q4_K decoded in MMVQ | not enough evidence for sidecar expansion |
| intermediate materializations | Gate, Up, SwiGLU/Q8 buffers | fused GLU result | materialization and launches are removable |

## Amdahl filter

Current recurrent Gate/Up is about `0.27 ms/layer × 48 = 12.96 ms/token`,
before attention-layer FFN work. Eliminating one projection's activation-side
loads and the separate SwiGLU materialization cannot remove its weight bytes,
so the conservative bound is roughly 2–5 ms/token; the upper bound including
reduction/launch and attention-layer FFN reuse is larger. This clears the
1–3 ms threshold and is the only currently confirmed llama.cpp structural
difference with that bound.

## Prior-art guard

`EXP-0139` already tested a dual Gate/Up projection launch that shared only the
input staging and was rejected (`-0.21%/-0.04%`). Repeating that design is out
of scope. The candidate below must also fuse the SwiGLU and Down-input Q8
boundary, so its expected value comes from removing intermediate vectors and
one complete activation pass.

## Decision

**SELECT** one bounded candidate: a Q4_K×Q8_1 fused Gate+Up+SwiGLU→Q8 kernel
for the exact Qwen3.8 FFN shape. Preserve the existing separate path as the
control. Do not add a generic fusion framework.

The candidate must first pass the existing replay and CTest contract, then
receive one TG64/TG128 A/B. If it fails the 3% whole-token threshold, reject
the fusion family and return to the differential.

## Re-evaluation

EXP-0170 tested the true fused Gate+Up+SwiGLU boundary. Replay passed, but TG64
fell 8.37%. The simple MIInfer kernel still decoded both Q4_K streams
independently and paid additional register pressure; removing the output
materialization was insufficient. The prototype was removed. B47 and B75 now
close the naive Gate/Up fusion family; any future attempt requires a different
weight/activation tiling mechanism, not another combined launch.
