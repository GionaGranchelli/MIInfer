# EXP-0065 — M6-A21 Qwen3.8-27B GPU hybrid block

## Status

KEEP — layers 0–3 compose on the real MI50 through the complete layer-3
boundary at positions 0 and 1; longer stateful composition remains deferred.

## Question

Can three validated recurrent GPU layers feed the validated full-attention GPU
layer while preserving persistent recurrent and KV state and the external
layer-output correctness envelope?

## Artifact and reference

* model: `/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf`
* SHA-256: `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
* fixture: `/tmp/m6a1-qwen38-reference`
* reference: llama.cpp commit `c0bc8591e8815c63cb01dd3f051a8b0df02501c9`
* device: MI50-class gfx906

## Candidate

Added `miinfer-m6a21-qwen35-gpu-hybrid-block`. It allocates each recurrent
layer's weights, workspace, convolution history, and `[48,128,128]` state once,
then executes:

```text
layer 0 recurrent → layer 1 recurrent → layer 2 recurrent → layer 3 full attention
```

The existing qwen35 GPU primitives are reused. The harness copies only the
fixture input and diagnostic layer outputs; recurrent and KV state remain on
the device between positions.

## Command

```bash
cmake --preset mi50-release
cmake --build --preset mi50-release \
  --target miinfer-m6a21-qwen35-gpu-hybrid-block -j2
build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  /tmp/m6a1-qwen38-reference
```

## Results

```text
position=0 layer=0 max_abs=0.000972833 rmse=0.00022218
position=0 layer=1 max_abs=0.00502329 rmse=0.000910132
position=0 layer=2 max_abs=0.0239334 rmse=0.00188165
position=0 layer=3 max_abs=0.0321732 rmse=0.0025433
position=1 layer=0 max_abs=0.00151062 rmse=8.82693e-05
position=1 layer=1 max_abs=0.00633049 rmse=0.00079827
position=1 layer=2 max_abs=0.0073587 rmse=0.00211816
position=1 layer=3 max_abs=0.0739498 rmse=0.00305624

max_error=0.0739498
M6-A21 qwen35 GPU hybrid block PASS
```

## Checks

* Release HIP target build: PASS
* real MI50 execution: PASS
* recurrent layers 0–2 compose into full-attention layer 3: PASS
* persistent recurrent state and layer-3 KV cache across positions 0→1: PASS
* layer-output checkpoints for all four layers: PASS

## Decision

**KEEP / M6-A21 complete.** The first GPU hybrid block is externally checked
through the full layer-3 output boundary. No end-to-end Qwen3.8 generation or
throughput claim is made yet; the next gate is longer stateful composition and
then the full 64-layer GPU trunk.

## Follow-up

Extend the block audit through positions 2, 4, 8, and 16, add state
fingerprints/reset-replay checks to the GPU path, then climb the composition
ladder.
