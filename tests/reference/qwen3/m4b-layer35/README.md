# M4-B9 external layer-35 reference trace

This fixture is an independent teacher-forced internal trace for terminal
layer 35 of the pinned Qwen3-8B Q4_0 model. It was captured from
`milpster/gfx906-llama-cpp` at commit
`6e4ef6c1a553b8f61ad77bba18e9ca05aa677295` with the canonical CPU settings:

```text
model: Qwen3-8B-q4_0-b968826d.gguf
model sha256: 458634762bea7dbe19f3ce0614465bafd15ee90e815229c427043afcf195d628
token: 14990
position: 0
flags: -ngl 0 -t 24 -tb 24
layer: 35
```

`layer-input.f32` is the same-run external `l_out-34` tensor. The terminal
`layer-output.f32` is the same tensor as `final-norm-input.f32`: the callback
renamed the terminal residual only for the explicit final-normalization
boundary check. All files are little-endian F32 in the reference logical
order.

The internal labels are intentionally narrow and correspond to the existing
`Qwen3LayerTrace` fields:

| File | Reference tensor | Elements |
| --- | --- | ---: |
| `attn-norm.f32` | `attn_norm-35` | 4096 |
| `q-projection.f32` | `Qcur_projection-35` | 4096 |
| `q-reshape.f32` | `Qcur_reshape-35` | 4096 |
| `q-normed.f32` | `Qcur_normed-35` | 4096 |
| `q-rope.f32` | `Qcur_rope-35` | 4096 |
| `v-projection.f32` | `Vcur_projection-35` | 1024 |
| `k-projection.f32` | `Kcur_projection-35` | 1024 |
| `k-reshape.f32` | `Kcur_reshape-35` | 1024 |
| `k-normed.f32` | `Kcur_normed-35` | 1024 |
| `k-rope.f32` | `Kcur_rope-35` | 1024 |
| `attention-output.f32` | `kqv_out-35` | 4096 |
| `o-projection.f32` | `o_projection-35` | 4096 |
| `ffn-input.f32` | `ffn_inp-35` | 4096 |
| `ffn-norm.f32` | `ffn_norm-35` | 4096 |
| `gate.f32` | `ffn_gate-35` | 12288 |
| `up.f32` | `ffn_up-35` | 12288 |
| `swiglu.f32` | `ffn_swiglu-35` | 12288 |
| `ffn-output.f32` | `ffn_down-35` | 4096 |
| `layer-output.f32` | terminal residual / final-norm input | 4096 |

Two independent captures using the same temporary reference instrumentation
were byte-identical for these tensors. The pinned reference checkout was not
modified; capture-only callback labels lived in a separate temporary
worktree.
