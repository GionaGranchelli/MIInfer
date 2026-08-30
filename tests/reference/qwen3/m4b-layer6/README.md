# M4-B6 external layer-6 reference trace

This fixture is an independent teacher-forced internal trace captured from
the pinned `milpster/gfx906-llama-cpp` reference at commit
`6e4ef6c1a553b8f61ad77bba18e9ca05aa677295`.

The capture used the pinned Qwen3-8B Q4_0 artifact:

```text
Qwen3-8B-q4_0-b968826d.gguf
SHA256 458634762bea7dbe19f3ce0614465bafd15ee90e815229c427043afcf195d628
```

The input is the explicit token `14990` at position `0`; the trace is from
the external CPU reference (`-ngl 0`).  `layer-input.f32` is the external
`l_out-5` captured in the same run as the layer-6 tensors.  MIInfer
teacher-forced replay uses that file as the layer-6 input, avoiding any
cross-capture numerical drift.

Each file is little-endian F32 in the reference's logical tensor order:

| File | Reference graph tensor | Shape |
| --- | --- | --- |
| `layer-input.f32` | `l_out-5` | 4096 |
| `attn-norm.f32` | `attn_norm-6` | 4096 |
| `q-projection.f32` | first `Qcur-6` | 4096 |
| `q-reshape.f32` | second `Qcur-6` | 128 × 32 |
| `q-normed.f32` | `Qcur_normed-6` | 128 × 32 |
| `q-rope.f32` | third `Qcur-6` | 128 × 32 |
| `v-projection.f32` | first `Vcur-6` | 1024 |
| `v-reshape.f32` | second `Vcur-6` | 128 × 8 |
| `k-projection.f32` | first `Kcur-6` | 1024 |
| `k-reshape.f32` | second `Kcur-6` | 128 × 8 |
| `k-normed.f32` | `Kcur_normed-6` | 128 × 8 |
| `k-rope.f32` | third `Kcur-6` | 128 × 8 |
| `ffn-input.f32` | `ffn_inp-6` | 4096 |
| `ffn-norm.f32` | `ffn_norm-6` | 4096 |
| `gate.f32` | `ffn_gate-6` | 12288 |
| `up.f32` | `ffn_up-6` | 12288 |
| `swiglu.f32` | `ffn_swiglu-6` | 12288 |
| `ffn-output.f32` | `ffn_out-6` | 4096 |
| `layer-output.f32` | `l_out-6` | 4096 |

The capture was made in a temporary reference worktree.  The pinned
reference checkout itself remains unchanged.  This fixture is diagnostic
evidence for M4-B6, not a performance result.
