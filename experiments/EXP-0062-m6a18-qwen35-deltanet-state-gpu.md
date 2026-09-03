# EXP-0062 — M6-A18 Qwen3.8-27B DeltaNet GPU state core

## Status

KEEP — real MI50 recurrent state-update primitive validated; complete qwen35
GPU layer and end-to-end performance remain deferred.

## Question

Can the Qwen3.8 Gated DeltaNet recurrent state update execute with persistent
state resident on the MI50 and reproduce the pinned external state/output
checkpoints across consecutive positions?

## Baseline and artifact

* model: `/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf`
* SHA-256: `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
* fixture: `/tmp/m6a1-qwen38-reference`
* reference: llama.cpp commit `c0bc8591e8815c63cb01dd3f051a8b0df02501c9`
* device: MI50-class gfx906, `stable_peak` policy

## Candidate

Added `launch_qwen35_deltanet_state_update`, a gfx906 HIP primitive with one
workgroup per value head. It updates the `[48,128,128]` FP32 recurrent state
in place and emits the `[48,128]` recurrent output. Query/key head sharing,
decay, beta update, and the final `1/sqrt(128)` scale follow the validated
host contract. The state allocation is persistent for both positions; only
inputs and checkpoint copies cross the host boundary in this diagnostic.

The existing qwen3 production executor is unchanged. This is deliberately a
recurrent-core slice, not a claim of a complete qwen35 layer.

## Command

```bash
cmake --preset mi50-release
cmake --build --preset mi50-release \
  --target miinfer-m6a18-qwen35-deltanet-state-gpu -j2
build/mi50-release/miinfer-m6a18-qwen35-deltanet-state-gpu \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  /tmp/m6a1-qwen38-reference
```

## Results

```text
position=0 output_max_abs=4.47035e-08 state_max_abs=1.19209e-06
position=1 output_max_abs=3.27826e-07 state_max_abs=1.96695e-06
state_elements=786432 persistent_state_bytes=3145728
M6-A18 qwen35 DeltaNet GPU state core PASS
```

Both positions passed the existing `1e-3` recurrent-output and `1e-2`
state-error envelopes. The state remains GPU-resident between position 0 and
position 1.

## Checks

* Release HIP target build: PASS
* real MI50 fixture execution: PASS
* positions 0→1 stateful GPU update: PASS
* `git diff --check`: PASS

## Decision

**KEEP / M6-A18 recurrent GPU core complete.** This proves the first real
Qwen3.8 recurrent state transition on gfx906. It does not yet validate GPU
RMSNorm, recurrent projections, convolution history, recurrent output
projection, FFN composition, or full-model generation.

## Follow-up

Add the remaining recurrent layer operations around this persistent state
core, beginning with GPU Qwen35 Q4_K/Q6_K projections and the four-token
convolution history. Keep the state-update primitive externally checked before
composing the complete layer.
