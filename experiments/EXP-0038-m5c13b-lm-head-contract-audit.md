# EXP-0038 — M5-C13b LM-head contract audit

**Status:** CLOSED — no valid apples-to-apples differential; no production change  
**Milestone:** M5  
**Date:** 2026-09-01  
**Model:** Qwen3-8B Q4_0 (`Qwen3-8B-q4_0-b968826d.gguf`)

## 1. Question

Can MIInfer's Qwen3 LM-head Q6_K×Q8_K GEMV be compared directly with the
pinned gfx906 llama.cpp quantized matvec at the same shape and contract?

## 2. MIInfer production path

The accepted MIInfer path is implemented by
`launch_qwen3_q6_k_q8_k_gemv`. Its input is `Q8KDeviceBlock` and its output is
FP32 logits. At P1, the C13a production audit measured:

```text
LM head: 2.942877 ms, 1 dispatch
Final norm→Q8_K: 0.008 ms, 1 dispatch
```

The live stable-peak recheck measured 2.935678 ms for the LM-head stage. The
production path and its Q8_K representation were unchanged.

## 3. External contract audit

The pinned llama.cpp gfx906 MMVQ implementation does not expose a Q6_K×Q8_K
GPU primitive. In `ggml/src/ggml-cuda/mmvq.cu`, the MMVQ function pointer is
declared with `const block_q8_1 *` input, and the Q6_K dispatch maps to
`vec_dot_q6_K_q8_1`. The corresponding implementation in `vecdotq.cuh` is
`vec_dot_q6_K_q8_1` and consumes `block_q8_1` blocks.

Therefore the available external path is:

```text
Q6_K × Q8_1 → FP32
```

while MIInfer's LM-head path is:

```text
Q6_K × Q8_K → FP32
```

These are different activation block formats, metadata, and dot-product
contracts. A direct latency comparison would confound kernel implementation
with the input representation and would not answer C13b's question.

## 4. Whole-token context control

For context only, the pinned external build measured the same model at
stable-peak clocks with the exact PP1/TG64 workload:

```text
llama.cpp TG64: 90.446439 tok/s
samples: 89.9673, 90.7383, 90.6337 tok/s
```

This is a whole-token control, not an LM-head measurement and is not used to
attribute an LM-head differential.

## 5. Decision

```text
C13b exact Q6_K×Q8_K differential: not measurable with the pinned external
GPU contract
Production LM-head path: unchanged
LM-head replacement: not selected
```

The source audit is a useful negative result: llama.cpp's available gfx906
MMVQ path cannot serve as the exact Q8_K LM-head oracle. Do not add a MIInfer
Q8_1 path solely to force a comparison. A future direct differential requires
an external Q8_K-compatible GPU kernel or a separately justified MIInfer
Q8_1 control; neither is part of C13b.

## 6. Follow-up

Return to the remaining fixed-floor ranking and select the next target only
from a valid same-contract measurement. The broad quantization,
normalization, and conversion categories remain candidates; no C13c is
preselected by this audit.
