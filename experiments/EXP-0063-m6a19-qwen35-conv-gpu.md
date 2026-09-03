# EXP-0063 — M6-A19 Qwen3.8-27B convolution GPU path

## Status

KEEP — real MI50 recurrent convolution/activation/split primitive validated;
complete qwen35 GPU layer remains deferred.

## Question

Can the four-tap Qwen3.8 recurrent convolution history remain on the GPU while
producing the expected SiLU-activated Q/K/V streams across consecutive
positions?

## Artifact and reference

* model: `/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf`
* SHA-256: `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
* fixture: `/tmp/m6a1-qwen38-reference`
* reference: llama.cpp commit `c0bc8591e8815c63cb01dd3f051a8b0df02501c9`

## Candidate

Added `launch_qwen35_conv_silu_split`, which updates a persistent four-slot
circular history of `[10240]` raw QKV vectors, applies the model's four-tap
convolution and SiLU, and writes separate Q/K/V buffers. Added a head-wise
L2-normalization primitive for the recurrent Q/K streams.

The existing qwen3 production path is unchanged. This is a diagnostic
operation slice; it uses fixture QKV projections and does not claim a complete
recurrent layer.

## Command

```bash
cmake --preset mi50-release
cmake --build --preset mi50-release \
  --target miinfer-m6a19-qwen35-conv-gpu -j2
build/mi50-release/miinfer-m6a19-qwen35-conv-gpu \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  /tmp/m6a1-qwen38-reference
```

## Results

```text
position=0 conv_silu_max_abs=1.19209e-07 query_l2_max_abs=5.96046e-08 key_l2_max_abs=5.96046e-08
position=1 conv_silu_max_abs=9.53674e-07 query_l2_max_abs=8.9407e-08 key_l2_max_abs=5.96046e-08
history_bytes=163840
M6-A19 qwen35 convolution GPU PASS
```

## Checks

* Release HIP target build: PASS
* real gfx906 fixture execution: PASS
* positions 0→1 persistent-history validation: PASS
* `git diff --check`: PASS

## Decision

**KEEP / M6-A19 complete.** The GPU path now has externally checked recurrent
convolution history, SiLU, Q/K/V split, and Q/K normalization alongside the
validated DeltaNet state core. Recurrent projections, gate/beta/decay
preparation, output projection, FFN, and full-model generation remain.

## Follow-up

Add the recurrent Q4_K/Q6_K projection path and the small F32 beta/alpha
projections, then compose those inputs with the two validated recurrent
primitives into a complete single-layer diagnostic.
