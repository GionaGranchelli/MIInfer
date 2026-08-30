# M3 Minimal Qwen3-8B Runtime Scaffold

M3 establishes model recognition, validated tensor ownership, GPU residency,
and static planning. It does not execute a transformer layer or generate
tokens; those are M4 responsibilities.

## Supported artifact

The loader intentionally accepts only the pinned dense model contract:

```text
model: Qwen/Qwen3-8B
revision: b968826d9c46dd6066d109eabc6255188de91218
artifact: Qwen3-8B-q4_0-b968826d.gguf
sha256: 458634762bea7dbe19f3ce0614465bafd15ee90e815229c427043afcf195d628
```

The artifact uses Q4_0 for the embedding and projection tensors, F32 for
normalization tensors, and Q6_K for the single output tensor. Q6_K is an
artifact-level exception retained explicitly; it is not silently dequantized
or handled by the Q4 projection path.

## Parser boundary

`miinfer_model` provides a Linux mmap-backed GGUF v3 reader. It validates
header counts, metadata strings and arrays, tensor ranks/dimensions, known
quantization block sizes, alignment, offsets, ranges, duplicate names, and
integer overflow before exposing immutable tensor views.

The supported parser tensor types are F32, F16, Q4_0, Q4_1, Q5_0, Q5_1,
Q8_0, Q8_1, Q2_K, Q3_K_S, Q4_K, Q5_K, Q6_K, and Q8_K. The Qwen3 model
validator then rejects every type not required by the pinned artifact.
Malformed or unsupported input fails with a contract-specific error.

## Validated configuration

| Field | Value |
|---|---:|
| architecture | `qwen3` |
| layers | 36 |
| hidden size | 4096 |
| intermediate size | 12288 |
| attention heads | 32 |
| KV heads | 8 |
| head dimension | 128 |
| vocabulary | 151936 |
| context length | 40960 |
| RoPE theta | 1000000 |
| RMSNorm epsilon | 0.000001 |

The 399-tensor inventory is:

| Type | Count | Bytes |
|---|---:|---:|
| F32 | 145 | 1,232,896 |
| Q4_0 | 253 | 4,257,054,720 |
| Q6_K | 1 | 510,504,960 |
| **Total** | **399** | **4,768,792,576** |

The per-layer inventory includes attention norm, Q/K/V/O, Q/K normalization,
FFN norm, and gate/up/down tensors. The single output norm is F32; the output
matrix is Q6_K. Tensor descriptors use GGUF's stored `[K, M]` order while the
kernel records continue to use the project's logical `M × K` convention.

## GPU ownership and plan

`Qwen3GpuPlan` validates gfx906 before allocation, allocates one contiguous
4,768,792,576-byte MI50 weight arena with 256-byte tensor alignment, and
uploads every immutable tensor exactly once. The real-model acceptance run
completed in approximately 28 seconds and copied representative head/tail
samples back successfully for embeddings, K, FFN gate, FFN down, and output
norm tensors.

Static temporary buffer accounting reserves 63,488 bytes for the currently
known activation/Q8 shapes. No KV cache or execution buffers are allocated
yet. The plan records offsets for future reuse and does not perform dynamic
allocation or tensor-name lookup during execution.

## Static projection kernel selection

The plan resolves the M2 evidence once during construction:

| Projection | Logical shape | Selected kernel |
|---|---:|---|
| Q | 4096 × 4096 | `zero-point-128` |
| K | 1024 × 4096 | `zero-point-wave64` |
| V | 1024 × 4096 | `zero-point-wave64` |
| O | 4096 × 4096 | `zero-point-128` |
| Gate | 12288 × 4096 | `zero-point-128` |
| Up | 12288 × 4096 | `zero-point-128` |
| Down | 4096 × 12288 | `zero-point-256` |

These are plan metadata only in M3. No GEMV is launched by the model-info
utility.

## Validation command

The inspection utility validates the pinned hash, model contract, optional
gfx906 device, arena upload, and representative device-byte samples:

```bash
./build/mi50-release/miinfer-model-info \
  /path/to/Qwen3-8B-q4_0-b968826d.gguf --json
```

The accepted physical-MI50 JSON result is retained at:

```text
bench/results/M3-qwen3-runtime/model-info.json
```

The utility is an inspection/validation tool, not an inference CLI.
