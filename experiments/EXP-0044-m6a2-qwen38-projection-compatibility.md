# EXP-0044 — M6-A2 Qwen3.8 projection/kernel compatibility audit

## Question

Which Qwen3.8-27B Q4_K_M projections can use existing MIInfer primitives,
which need a contract-preserving adapter, and which require new gfx906 kernels?

## Artifact and method

The audit used the pinned artifact from EXP-0042/0043:

```text
/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
SHA-256: 7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169
```

The read-only reporter is `tools/m6a2_compatibility_audit.cpp`, built as
`miinfer-m6a2-compatibility-audit`. It groups the real GGUF tensors by layer
pattern and prints exact counts, types, dimensions, and bytes.

## Existing MIInfer contracts

The current production kernels provide:

* Q4_0 × Q8_1/Q8Exact GEMV, with existing Qwen3-specific output conversion;
* Q4_0 embedding lookup;
* Q6_K × Q8_K GEMV and Q8_K quantization;
* F32 RMSNorm, elementwise operations, head normalization, RoPE, and the old
  32-head/8-KV-head cached-attention path;
* old Qwen3 model planning for 36 dense layers, hidden 4096, FFN 12288, and
  vocabulary 151936.

There is no Q4_K×Q8, Q5_K×Q8, Q4_K embedding, DeltaNet state/update, or
Qwen35 hybrid-layer primitive.

## Compatibility map

| Qwen3.8 tensor/path | Count / shape / type | Activation or operation contract | Decision |
| --- | --- | --- | --- |
| Token embedding | 1 × `[5120,248320]` Q4_K | Q4_K row lookup → F32 | **NEW KERNEL REQUIRED** |
| Recurrent `attn_qkv` | 48 × `[5120,10240]` Q4_K/Q6_K | recurrent projection; not old Q4_0 GEMV | **NEW KERNEL REQUIRED** |
| Recurrent `attn_gate` | 48 × `[5120,6144]` Q4_K | recurrent gate projection | **NEW KERNEL REQUIRED** |
| Recurrent `ssm_out` | 48 × `[6144,5120]` Q5_K | recurrent output projection | **NEW KERNEL REQUIRED** |
| Recurrent SSM tensors | 48-layer F32 conv/state/gate tensors | DeltaNet convolution, state update, gating | **NEW KERNEL REQUIRED** |
| Full-attention Q | 17 × `[5120,12288]` Q4_K | 24 query heads, 4 KV heads | **NEW KERNEL REQUIRED** |
| Full-attention K | 17 × `[5120,1024]` Q4_K | 4 KV heads, head dimension 256 | **NEW KERNEL REQUIRED** |
| Full-attention V | 17 × `[5120,1024]` Q4_K/Q6_K | 4 KV heads, head dimension 256 | **NEW KERNEL REQUIRED** |
| Full-attention O | 17 × `[6144,5120]` Q4_K | hybrid attention output | **NEW KERNEL REQUIRED** |
| Full-attention Q/K norms | 17 each × `[256]` F32 | per-head L2 normalization | **ADAPT** existing normalization primitives |
| FFN Gate/Up | 65 each × `[5120,17408]` Q4_K | Q4_K activation projection | **NEW KERNEL REQUIRED** |
| FFN Down | 65 × `[17408,5120]` Q4_K/Q6_K | Q4_K generally; Q6_K subset can use Q6 path | **ADAPT** Q6 subset / **NEW** Q4_K |
| Final RMSNorm | 1 × `[5120]` F32 | F32 RMSNorm | **ADAPT** shape/plan only |
| LM head | 1 × `[5120,248320]` Q6_K | Q8_K-compatible candidate path | **ADAPT** loader/plan; kernel shape is compatible |
| NextN `eh_proj` | 1 × `[10240,5120]` Q8_0 | MTP-only projection | **NEW KERNEL REQUIRED** if MTP is enabled |

`ADAPT` means the existing primitive is conceptually compatible but the model
loader, dimensions, or execution plan must be changed. `NEW KERNEL REQUIRED`
means no current primitive consumes the GGUF weight/activation contract.

## Important findings

1. The selected artifact's 27B main path is overwhelmingly Q4_K/Q5_K/Q6_K,
   while current MIInfer's optimized projection path is Q4_0×Q8_1. The old
   Q4_0 GEMV results cannot be reused as a Qwen3.8 performance conclusion.
2. The existing Q6_K×Q8_K LM-head primitive is the only directly shape-
   compatible large projection candidate, subject to loader and external-logit
   validation.
3. Qwen3.8 has 48 recurrent layers and 16 full-attention layers; the old
   all-full-attention Qwen3 executor cannot be adapted by metadata changes
   alone.
4. Block 64/NextN is outside the initial text-generation path and should not
   block main-model bring-up.

## Decision

**KEEP / M6-A2 complete.** The compatibility boundary is now explicit. No
Qwen3.8 production execution code was changed.

## M6-A3 next task

Implement one real recurrent/DeltaNet layer using the EXP-0043 external
fixture. Start with layer 0 at position 0, validate the embedding, input norm,
recurrent projection/state/update/output, residual, and FFN checkpoints, then
extend the same layer to positions 1/2/4/8. Do not begin full-model execution
or performance optimization until the stateful single-layer external contract
passes.
