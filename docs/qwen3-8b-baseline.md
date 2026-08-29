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

Conversion status: NOT CREATED in this environment. The resulting F16 GGUF
size and SHA256 must be recorded here after conversion completes on a suitable
host. The GGUF is not committed to MIInfer.

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
