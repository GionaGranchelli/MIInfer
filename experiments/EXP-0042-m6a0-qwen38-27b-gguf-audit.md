# EXP-0042 — M6-A0 Qwen3.8-27B GGUF and architecture audit

**Status:** COMPLETE — architecture audit; no production implementation
**Milestone:** M6-A0
**Date:** 2026-09-02
**Repository commit:** `ed87fd85b71c55ddc624abee1a1c8a90ffd23c77`

## 1. Question

Which local Qwen3.8-27B GGUF should be the first M6 target, what exact
architecture and tensor contracts does it contain, and which parts of the
current MIInfer runtime can be reused?

## 2. Local artifacts

The requested LM Studio directory was not present:

```text
/home/fedora-workstation/.lmstudio/models/lmstudio-community/Qwen3.8-27B-GGUF
```

The available candidates were found under `/home/fedora-workstation/models/`:

| Artifact | Size | Quantized tensor mix | Decision |
|---|---:|---|---|
| `Qwen3.8-27B-Q3_K_M.gguf` | 13,818,690,528 bytes / 12.87 GiB | Q3_K_S, Q4_K, Q5_K, Q6_K | Not first target; lower-quality and a different kernel mix |
| `Qwen3.8-27B-Q4_K_M.gguf` | 17,106,775,008 bytes / 15.93 GiB | Q4_K, Q5_K, Q6_K, Q8_0 | **Selected** |
| `Qwen3.8-27B-UD-Q6_K.gguf` | 21,983,677,344 bytes / 20.47 GiB | Q6_K, Q5_K, Q8_0, IQ3_S, Q4_K | Not first target; heterogeneous UD format increases bring-up scope |

### Selected target

```text
/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
```

Q4_K_M is the best initial target because it fits comfortably within the
MI50's 32 GB VRAM, is the conventional quality/performance point, and avoids
making M6 simultaneously a mixed-quantization bring-up. The Q3 artifact can
be revisited for a lower-memory target; UD-Q6_K should wait until the base
execution path is proven.

## 3. GGUF metadata

The selected file is GGUF v3 with 866 tensors and 51 metadata entries.

| Property | Value |
|---|---:|
| `general.name` | `Qwen3.8-27B` |
| `general.architecture` | `qwen35` |
| `general.size_label` | `27B` |
| `qwen35.block_count` | 65 total blocks |
| `qwen35.nextn_predict_layers` | 1 |
| `qwen35.embedding_length` | 5120 |
| `qwen35.feed_forward_length` | 17408 |
| `qwen35.context_length` | 262144 |
| `qwen35.attention.head_count` | 24 |
| `qwen35.attention.head_count_kv` | 4 |
| `qwen35.attention.key_length` | 256 |
| `qwen35.attention.value_length` | 256 |
| `qwen35.attention.layer_norm_rms_epsilon` | approximately `1e-6` |
| `qwen35.rope.freq_base` | `10000000` |
| `qwen35.rope.dimension_count` | 64 |
| `qwen35.rope.dimension_sections` | `[11, 11, 10, 0]` |
| `qwen35.full_attention_interval` | 4 |
| `qwen35.ssm.conv_kernel` | 4 |
| `qwen35.ssm.state_size` | 128 |
| `qwen35.ssm.group_count` | 16 |
| `qwen35.ssm.inner_size` | 6144 |
| `qwen35.ssm.time_step_rank` | 48 |
| tokenizer model | `gpt2` |
| tokenizer pre-tokenizer | `qwen35` |
| vocabulary | 248,320 tokens |
| BOS / EOS / padding IDs | 248044 / 248046 / 248055 |

The file does not contain an explicit recurrent-layer array, so the layer
pattern is derived from `full_attention_interval` and tensor presence. This
matches the local llama.cpp qwen35 loader.

## 4. Layer pattern

The 64-layer main trunk is hybrid:

```text
main layers 0..63:
  recurrent gated-DeltaNet: 0,1,2, 4,5,6, ..., 60,61,62  (48 layers)
  full attention:           3,7,11,15, ..., 59,63       (16 layers)

appended block 64:
  dense full-attention NextN/MTP block, not part of the ordinary 64-layer
  trunk forward unless MTP is explicitly executed
```

The recurring trunk pattern is therefore:

```text
DeltaNet, DeltaNet, DeltaNet, full attention
```

The local llama.cpp source confirms that `n_layer_all=65`,
`n_layer_nextn=1`, and `n_layer()=64`; its main graph executes only the first
64 blocks and its MTP graph handles block 64 separately.

### Recurrent layer operations

Each recurrent layer contains the model-specific tensors for:

* attention normalization;
* mixed Q/K/V projection and attention gate projection;
* convolution state and convolution weights;
* delta-net alpha, beta, A, time-step bias, and state normalization;
* recurrent output projection;
* dense FFN gate, up, and down projections;
* post-attention normalization.

The reference graph performs QKV/gate projections, beta sigmoid, alpha plus
time-step bias and softplus, gated delta-net convolution/state update,
L2-normalized Q/K, recurrent attention, gated normalization, output projection,
then the dense FFN and residuals.

### Full-attention layer operations

Each main full-attention layer contains:

* attention normalization;
* joint query-plus-gate projection;
* Q/K normalization;
* K/V projections;
* sectioned RoPE;
* cached attention with four KV heads;
* sigmoid attention gate and output projection;
* post-attention normalization;
* dense FFN and residuals.

Block 64 has the same dense full-attention/FFN tensor set plus these present
NextN tensors:

```text
nextn.eh_proj.weight
nextn.enorm.weight
nextn.hnorm.weight
nextn.shared_head_norm.weight
```

The optional `nextn.embed_tokens.weight` and
`nextn.shared_head.head.weight` tensors are absent from this artifact.

## 5. Tensor inventory

The selected Q4_K_M artifact has 866 tensors:

| GGUF type | Tensor count | Accounted bytes | Approx. size |
|---|---:|---:|---:|
| F32 | 456 | 116,055,008 | 0.108 GiB |
| Q4_K | 294 | 11,370,332,160 | 10.589 GiB |
| Q5_K | 48 | 1,038,090,240 | 0.967 GiB |
| Q6_K | 67 | 4,526,592,000 | 4.216 GiB |
| Q8_0 | 1 | 55,705,600 | 0.052 GiB |
| **Total** | **866** | **17,106,775,008** | **15.932 GiB** |

The three global tensors are `token_embd.weight`, `output_norm.weight`, and
`output.weight`. Tensor grouping is:

* 48 recurrent main-trunk layers × 14 tensors = 672;
* 16 full-attention main-trunk layers × 11 tensors = 176;
* block 64 × 15 tensors = 15;
* 3 global tensors;
* total = 866.

Representative selected-artifact contracts include:

| Tensor | Shape | Type |
|---|---|---|
| `token_embd.weight` | `[5120, 248320]` | Q4_K |
| `output.weight` | `[5120, 248320]` | Q6_K |
| `blk.0.attn_qkv.weight` | `[5120, 10240]` | Q6_K |
| `blk.0.attn_gate.weight` | `[5120, 6144]` | Q4_K |
| `blk.0.ssm_out.weight` | `[6144, 5120]` | Q5_K |
| `blk.0.ffn_gate/up.weight` | `[5120, 17408]` | Q4_K |
| `blk.0.ffn_down.weight` | `[17408, 5120]` | Q6_K |
| `blk.3.attn_q.weight` | `[5120, 12288]` | Q4_K |
| `blk.3.attn_k/v.weight` | `[5120, 1024]` | Q4_K / Q6_K |
| `blk.64.nextn.eh_proj.weight` | `[10240, 5120]` | Q8_0 |

The artifact has no Q4_0, Q8_1, or Q8_K weight tensors. Existing Qwen3-8B
Q4_0 kernels therefore cannot be treated as direct model execution support.

## 6. Current MIInfer compatibility

### Unsupported today

The current loader rejects this artifact immediately because it requires
`general.architecture=qwen3`; this file declares `qwen35`. It also hard-codes
the previous model's 36-layer/4096-hidden/12288-FFN/32-head/8-KV-head/151936
vocabulary contract.

The current runtime has no production path for:

* qwen35 metadata and tensor naming;
* 48-layer gated DeltaNet/linear-attention execution;
* SSM convolution state;
* recurrent delta-net state read/update/rollback;
* mixed QKV and attention-gate projections;
* sectioned/partial RoPE configuration;
* Qwen3.8's 24/4 attention geometry;
* Q4_K, Q5_K, and Q6_K projection kernels for the new shapes;
* Q8_K/Q8_0-compatible activation contracts for these projections;
* Q4_K token embedding;
* qwen35 tokenizer pre-tokenizer (`qwen35` is currently rejected; only
  `gpt2` + `qwen2` is accepted);
* NextN/MTP block execution;
* Qwen3.8 full-model stateful generation and external-reference fixtures.

The existing static GPU workspace is also hard-coded to 4096/12288-sized
buffers and cannot be reused unchanged.

### Reusable pieces

These are candidates for reuse after model-specific validation, not claims of
drop-in compatibility:

* mmap-backed GGUF parsing and the GGUF K-quant type registry;
* gfx906 device validation and contiguous GPU weight-arena ownership;
* static tensor planning and weight-integrity checks;
* F32 RMSNorm reduction and elementwise primitives;
* F32/F16 conversion primitives;
* existing Q6_K GEMV machinery as a possible starting point for the Qwen3.8
  Q6_K projections and output head, subject to activation and shape contracts;
* existing GPU argmax for the 248,320-element vocabulary;
* existing persistent workspace/lifetime approach;
* existing cooperative full-attention ideas, after adapting the Qwen3.8
  head/cache/RoPE contract;
* existing tokenizer BPE storage/merge machinery, after adding and validating
  qwen35 pre-tokenization semantics.

The existing Q4_0×Q8_1 GEMV, Q4_0 embedding, Qwen3 layer executor, old KV
cache, and old Qwen3 model config are not direct reuse candidates for the
selected artifact.

## 7. Estimated MI50 VRAM budget

These are planning estimates before an M6 workspace is implemented:

| Component | Estimate |
|---|---:|
| Q4_K_M source file / accounted weight arena | 15.93 / approximately 15.94 GiB |
| Recurrent state, 48 × 6144 × 128 F32 | 0.141 GiB |
| Recurrent convolution state, 48 layers, F32 | 0.006 GiB |
| Main full-attention KV cache, F16, 16 layers | 0.488 GiB at 8K; 1.953 GiB at 32K; 16 GiB at 262K |
| Initial activation/workspace reserve | approximately 0.5–1.0 GiB |
| Final FP32 logits | approximately 0.001 GiB |

Approximate total is therefore 17.1–17.6 GiB at 8K context and about
19.0–19.5 GiB at 32K, before allocator/driver headroom. A full 262K F16 KV
cache would exceed a comfortable 32 GB deployment budget once weights and
workspace are included. M6 should choose an explicit practical context cap
and measure the real state/cache representation rather than promise the
training context length.

The recurrent-state estimate follows the reference formula
`n_embd_s = ssm_state_size × ssm_inner_size = 128 × 6144` elements per
recurrent layer. The convolution-state estimate uses
`(conv_kernel - 1) × (ssm_inner_size + 2 × group_count × state_size)` elements
per recurrent layer. Actual M6 cache/state dtypes may reduce these figures,
but the initial planner should budget conservatively.

## 8. Reference implementation

The initial external reference is the local upstream llama.cpp checkout:

```text
path:   /home/fedora-workstation/llama.cpp
commit: c0bc8591e8815c63cb01dd3f051a8b0df02501c9
binary: /home/fedora-workstation/llama.cpp/build/bin/llama-cli
```

The qwen35 architecture loader and graph are in `src/models/qwen35.cpp`; the
tokenizer support is in `src/llama-vocab.cpp`. The reference explicitly
implements the hybrid recurrent/full-attention graph, qwen35 tensor mapping,
and NextN/MTP separation. It is the initial behavioral and architecture
reference, not a requirement to reproduce llama.cpp's internal graph or
intermediate bytes.

The attempted local `llama-cli` model-load check was interrupted after a
bounded wait while it was still showing model-load progress. Therefore this
audit records source-level reference support and binary/version identity; it
does not claim a completed reference generation run.

## 9. Exact proposed M6-A1 task

Create a pinned external tensor-reference fixture for the selected
`Qwen3.8-27B-Q4_K_M.gguf` and the local llama.cpp commit above.

The fixture task must:

1. pin the exact llama.cpp revision and selected GGUF path/identity;
2. pin deterministic token IDs and a short prompt;
3. export checkpoints for embedding output, one recurrent DeltaNet layer, one
   full-attention layer, one complete four-layer hybrid block, final norm, and
   LM-head logits;
4. export recurrent state and full-attention KV state at positions 1, 2, 4,
   8, 16, 32, and 64 where the reference exposes them;
5. record tensor shapes, dtypes, positions, cache/state reset behavior, and
   tokenizer output;
6. add a MIInfer-side fixture reader/comparison utility without changing
   production execution;
7. define tolerances for F32/F16 and quantized boundaries, including top-1,
   top-k, and logit-error checks;
8. verify that the fixture can be regenerated and that its provenance is
   documented in a new experiment record.

The A1 exit gate is an independently reproducible reference bundle that a
future MIInfer operation/layer implementation can consume without relying on
the old Qwen3-8B trajectory.

## 10. Files changed

```text
experiments/EXP-0042-m6a0-qwen38-27b-gguf-audit.md
```

No production source, build configuration, external checkout, model artifact,
or benchmark behavior was changed.

## 11. Checks run

* Enumerated all three local `Qwen3.8-27B-*.gguf` artifacts and recorded byte
  sizes.
* Parsed the selected GGUF header, metadata, tensor names, dimensions, raw
  tensor types, and offsets using a read-only mmap audit.
* Confirmed selected tensor type counts: F32 456, Q4_K 294, Q5_K 48,
  Q6_K 67, Q8_0 1.
* Confirmed main full-attention tensor indices and recurrent QKV tensor
  indices from the selected artifact.
* Inspected local llama.cpp qwen35 loader/graph and recurrent-memory formulas.
* Recorded local llama.cpp revision `c0bc8591e8815c63cb01dd3f051a8b0df02501c9`
  and `llama-cli --version` output (`1 (c0bc859)`).
* Confirmed current MIInfer loader hard-codes the previous qwen3 model and
  tokenizer contracts; the current model-info binary is not a Qwen3.8 audit
  tool and was not used to reinterpret the artifact.
* Ran `git diff --check` after the documentation change.
* Working tree was clean before this experiment file was added.

## 12. Conclusion

Qwen3.8-27B Q4_K_M is the correct first M6 artifact. It fits the MI50 memory
budget at practical contexts, but it is a genuinely new `qwen35` hybrid model,
not a larger instance of the existing Qwen3 path. The first implementation
work must therefore be reference fixtures and tensor/operation bring-up for
gated DeltaNet plus full attention; no performance kernel selection is
justified before that contract is pinned.
