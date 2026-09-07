# EXP-0208 — Full-Attention Projection Batching

## Hypothesis

The layer-major prefill path batches recurrent projections but still executes
all full-attention Q/K/V projections one token at a time. Reusing the existing
validated B=4 native kernels for combined Q+gate+K and V should reduce the
remaining prompt-processing cost without changing causal KV or attention state
transitions.

## Baseline

Default token-major execution with the same model, prompt, and graph setting:

```text
MIINFER_HIP_GRAPH=0
MIINFER_PREFILL_LAYER_MAJOR=0
model: Qwen3.8-27B-Q4_K_M.gguf
prompt: 512 repeated `hello` tokens
```

## Candidate

The opt-in layer-major path now also prepares, per four-token chunk, the
full-attention layer's:

- combined Q+gate+K projection;
- native V projection;
- normalized Q8_1 input shared by both projections.

The scalar per-token path still performs RoPE, KV-cache writes, attention,
output projection, and FFN in position order. Unsupported attention layouts
fall back to the existing scalar projection path.

## Environment

- GPU: gfx906 (MI50/MI60-visible device)
- SCLK: 1606 MHz
- HBM/MCLK: 1000 MHz
- ROCm: 6.4.0 / LLVM 20
- model: `Qwen3.8-27B-Q4_K_M.gguf`
- graph capture: disabled for the A/B comparison

## Correctness

The candidate and baseline selected the same first token (`hello`) at P16,
P128, and P512. At P128 with 16 generated tokens, both produced the same
16-token `hello` continuation. Decode remained on the existing scalar path.

## Results

Three prefill-only P512 repetitions per side:

| Path | Runs (ms) | Median | Throughput |
| --- | --- | ---: | ---: |
| Default token-major | 15,830.36 / 15,869.06 / 15,907.98 | 15,869.06 ms | 32.20 tok/s |
| Layer-major + attention batching | 12,811.92 / 12,812.97 / 12,800.25 | 12,812.97 ms | 39.96 tok/s |

The candidate improves P512 prefill throughput by approximately 24.1%.
The opt-in buffers add approximately 4 MiB across the 16 full-attention
layers.

## Decision

**KEEP as an opt-in production candidate.** It is correct for the tested
short and longer-generation checks and produces a repeatable improvement, but
it does not meet the M11-B target of 100 tok/s. The environment remains
disabled by default until the full layer-major path's longer-generation
correctness qualification and the target gate are complete.

## Follow-up

Profile the remaining recurrent projection/tail stages before adding another
batch size or new kernel mapping. Any larger-batch attempt must first pass a
small independent GPU correctness check; EXP-0207 demonstrated the cost of
using a model run as the first test for an unvalidated LDS mapping.
