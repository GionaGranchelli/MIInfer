# Qwen3-8B Dense Control Model

MIInfer's initial dense control is the official `Qwen/Qwen3-8B` model. This
document freezes model metadata for kernel-shape experiments; model weights are
not stored in this repository.

## Source

```text
Repository/model: Qwen/Qwen3-8B
Exact source revision: b968826d9c46dd6066d109eabc6255188de91218
Download date: 2026-08-29
Configuration source: https://huggingface.co/Qwen/Qwen3-8B/resolve/b968826d9c46dd6066d109eabc6255188de91218/config.json
Original tensor dtype: bfloat16
Model architecture: Qwen3ForCausalLM / qwen3
License: Apache-2.0
```

The revision is the Hugging Face model commit returned for the official model
as of the download date. Reproducible downloads must pass this revision to the
Hub client; do not use an unpinned `main` checkout.

## Exact configuration

| Field | Value |
| --- | ---: |
| `hidden_size` | 4096 |
| `intermediate_size` | 12288 |
| `num_hidden_layers` | 36 |
| `num_attention_heads` | 32 |
| `num_key_value_heads` | 8 |
| `head_dim` | 128 |
| `vocab_size` | 151936 |
| `torch_dtype` | `bfloat16` |
| `attention_bias` | `false` |
| `hidden_act` | `silu` |
| `rope_theta` | 1000000 |

## F16 artifact route

The reproducible source snapshot is downloaded outside the repository:

```bash
hf download Qwen/Qwen3-8B \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --local-dir /path/to/qwen3-8b-b968826d
```

The pinned external reference converter is
`milpster/gfx906-llama-cpp` at commit
`6e4ef6c1a553b8f61ad77bba18e9ca05aa677295`.
The source snapshot is 16,381,470,720 bytes according to the pinned Hub
revision metadata. Its download was attempted on 2026-08-29 but was stopped
before completion because the unauthenticated transfer rate was not suitable
for this run.
The exact conversion command on a host with the completed snapshot is:

```bash
python3 /path/to/gfx906-llama-cpp/convert_hf_to_gguf.py \
  /path/to/qwen3-8b-b968826d \
  --outfile /path/to/artifacts/Qwen3-8B-f16.gguf \
  --outtype f16
```

Conversion completed on 2026-08-29 with the command above. The artifact is
stored outside the repository at
`/home/fedora-workstation/Development/mi50-artifacts/Qwen3-8B-f16-b968826d.gguf`:

```text
size: 16388044192 bytes
sha256: c1fd1fc17831ebc0001d81c97a3f78626dd1f977841dec532eef60177abb2a1c
source shards: all five expected safetensors files present
```

Source shard SHA256 values:

```text
model-00001-of-00005.safetensors  31d6a825ae35f11fb85b195b4c42c146c051e446433125a215336abdf95cbf5f
model-00002-of-00005.safetensors  5991236cea6fe21f3d43cab0f0e84448734fbbe0789816202989f2ddc9d18282
model-00003-of-00005.safetensors  c5185c4794be2d8a9784d5753c9922db38df478ce11f9ed0b415b7304d896836
model-00004-of-00005.safetensors  b5ee7de71fbf17db3d5704e0c8f2bc7d005ca9e1d7ca2aeb19827b0cfcaa917a
model-00005-of-00005.safetensors  20c2d6366ab85c90786ccdd829cd2b9e7d30ef3b2ebbb998280e7e4014b542ff
```

The GGUF is not committed to MIInfer.

Reference smoke status: BLOCKED. The pinned CLI loaded the GGUF metadata but
reported `ggml_cuda_init: failed to initialize ROCm: no ROCm-capable device is
detected`; `--gpu-layers` was ignored and the attempted run was stopped rather
than accepted as a CPU fallback. No GPU generation result is claimed.

## Real model projection shapes

Shapes use the GEMV convention `M x K`: one input vector of length `K` times
a weight matrix producing `M` output values. Every entry below is a real
Qwen3-8B shape, not a placeholder.

| Projection | Matrix shape (`M x K`) | Derivation |
| --- | ---: | --- |
| Q projection | `4096 x 4096` | `num_attention_heads * head_dim` by `hidden_size` |
| K projection | `1024 x 4096` | `num_key_value_heads * head_dim` by `hidden_size` |
| V projection | `1024 x 4096` | `num_key_value_heads * head_dim` by `hidden_size` |
| Output projection | `4096 x 4096` | `hidden_size` by `num_attention_heads * head_dim` |
| FFN gate projection | `12288 x 4096` | `intermediate_size` by `hidden_size` |
| FFN up projection | `12288 x 4096` | `intermediate_size` by `hidden_size` |
| FFN down projection | `4096 x 12288` | `hidden_size` by `intermediate_size` |

These shapes feed EXP-0002. No GEMV kernel or model loader is implemented by
this metadata record.
