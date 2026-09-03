# EXP-0064 — M6-A20 Qwen3.8-27B recurrent layer on GPU

## Status

KEEP — complete layer-0 recurrent GPU vertical slice validated on real gfx906;
full hybrid-block and generation integration remain deferred.

## Question

Can the recurrent projections, beta/alpha preparation, convolution/state
primitives, recurrent output projection, and FFN compose into one externally
checked Qwen3.8 recurrent layer while keeping persistent state on the GPU?

## Artifact and reference

* model: `/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf`
* SHA-256: `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
* fixture: `/tmp/m6a1-qwen38-reference`
* reference: llama.cpp commit `c0bc8591e8815c63cb01dd3f051a8b0df02501c9`
* device: MI50-class gfx906

## Candidate

Added GPU Q5_K×Q8_K GEMV for `ssm_out`, F32 GEMV for beta/alpha, and a
beta/decay preparation kernel. Added
`miinfer-m6a20-qwen35-recurrent-layer-gpu`, which composes:

```text
attn RMSNorm
→ QKV and gate projections
→ beta/alpha projections and preparation
→ persistent convolution/SiLU/QKV split
→ persistent DeltaNet state update
→ recurrent normalization and gate
→ Q5_K recurrent output projection
→ attention residual
→ post-attention RMSNorm
→ Qwen FFN and residual
```

The Q5_K device view matches the GGUF byte layout. State and convolution
history remain device-resident between positions 0 and 1; host copies are
diagnostic checkpoint reads only.

## Command

```bash
cmake --preset mi50-release
cmake --build --preset mi50-release \
  --target miinfer-m6a20-qwen35-recurrent-layer-gpu -j2
build/mi50-release/miinfer-m6a20-qwen35-recurrent-layer-gpu \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  /tmp/m6a1-qwen38-reference
```

## Results

```text
position=0 attn_norm max_abs=0
position=0 qkv max_abs=3.8147e-06
position=0 recurrent_output max_abs=7.45058e-08
position=0 state max_abs=8.34465e-07
position=0 gated_recurrent max_abs=7.15256e-07
position=0 attention_residual max_abs=0.00163269
position=0 post_attention_norm max_abs=0.000246942
position=0 ffn_output max_abs=0.0018158
position=0 layer_output max_abs=0.000972833

position=1 attn_norm max_abs=0
position=1 qkv max_abs=7.62939e-06
position=1 recurrent_output max_abs=2.08616e-07
position=1 state max_abs=1.90735e-06
position=1 gated_recurrent max_abs=4.76837e-07
position=1 attention_residual max_abs=0.00129795
position=1 post_attention_norm max_abs=0.000231981
position=1 ffn_output max_abs=0.000371102
position=1 layer_output max_abs=0.00151062

max_error=0.0018158
M6-A20 qwen35 recurrent GPU layer PASS
```

The recurrent state is persistent across the two positions and matches the
external next-position state checkpoint within `2e-6`. The layer output stays
within the external GPU envelope without changing the reference or weakening
the comparison after execution.

## Checks

* Release HIP target build: PASS
* real MI50 execution: PASS
* positions 0→1 recurrent state and convolution history: PASS
* complete layer-0 output through FFN/residual: PASS
* `git diff --check`: PASS

## Decision

**KEEP / M6-A20 complete.** Qwen3.8 recurrent projections, beta/alpha
preparation, persistent recurrent state, recurrent output projection, and the
complete recurrent layer are now externally checked on gfx906. This is not
yet full Qwen3.8 inference: layers 0–3 GPU composition, final norm/LM head,
stateful generation, and M6-B1 performance remain.

## Follow-up

Compose this recurrent layer with the existing GPU full-attention layer into
the first four-layer hybrid-block audit, then climb the GPU composition ladder.
