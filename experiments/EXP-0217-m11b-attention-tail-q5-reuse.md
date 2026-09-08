# EXP-0217 — M11-B Deferred Attention Tail and Shape-Specific Q5 Reuse

## Hypothesis

The layer-major path already batches full-attention Q/K/V, but still runs the
output projection, residual normalization, and FFN token-by-token. Keeping
causal attention and KV writes ordered while deferring only that post-attention
tail should batch the remaining native Q4_K projections. The recurrent native
Q5_K `ssm_out` projection should likewise use a word-reuse B=4 mapping rather
than the rejected generic decoder-state batch kernel from EXP-0216.

## Candidate

For each four-token full-attention microtile:

1. run normalized Q/K/V, RoPE, causal attention, and KV writes in position order;
2. copy each gated attention result into bounded prefill workspace;
3. batch native Q4_K O, residual/post-normalization, paired SwiGLU, and FFN
   Down projections;
4. emit the four next-layer activations in order.

The Q5 candidate decodes each native weight word once and accumulates four
outputs, avoiding the generic B=4 decoder-state kernel rejected in EXP-0216.
Both paths are active by default only inside the opt-in layer-major prefill;
`MIINFER_PREFILL_ATTN_TAIL=0` and `MIINFER_PREFILL_SSM_BATCH=0` provide matched
A/B controls.

## Environment

- GPU: AMD Instinct MI60/MI50-visible gfx906
- SCLK/MCLK: 1606/1000 MHz
- ROCm: 6.4.0 / LLVM 20
- Build: `mi50-release`
- Model: `Qwen3.8-27B-Q4_K_M.gguf`
- Prompt: 513 repeated `hello` tokens for production-shape runs
- `MIINFER_PREFILL_LAYER_MAJOR=1`
- `MIINFER_HIP_GRAPH=0`
- Native Q4_K attention O, QKV, V, gate/up/down and native Q5_K `ssm_out`
  enabled

## Causal correctness

Q/K/V projection, RoPE, attention, and KV-cache writes remain in the existing
position-ordered `run()` path. Only the post-attention tail is deferred.
The P17 candidate produced the same 16-token `hello` continuation as the
existing layer-major control. A P128 multi-chunk run also produced the same
continuation as both candidate-disabled controls. The full P128 prompt
selected `<|im_start|><think>` in all compared paths; no claim of agreement
with token-major execution is made because the existing layer-major oracle has
its own numerical trajectory at this prompt.

## Results

Three release repetitions per side at P513:

| Path | Runs (ms) | Median | PP tok/s |
| --- | --- | ---: | ---: |
| Existing layer-major, both new tails disabled | 12861.62 / 12828.87 / 12857.99 | 12857.99 | 39.90 |
| Deferred attention tail, scalar Q5 `ssm_out` | 11879.54 / 11865.24 / 11869.04 | 11869.04 | 43.22 |
| Deferred attention tail + shape-specific Q5 B=4 | 11386.71 / 11395.72 / 11396.37 | 11395.72 | 45.02 |

The attention tail adds `+8.3%` over the matched layer-major control. The
shape-specific Q5 mapping adds another `+4.2%`, for `+12.8%` overall. A final
P128 candidate run measured `46.39 tok/s` prefill and `33.19 tok/s` decode.
P16 improved
from `40.45 tok/s` with both tails disabled to `45.41 tok/s` with both enabled.

The new attention workspace is approximately 20 MiB across 16 layers. It is
allocated once when layer-major prefill is enabled and is not used by decode.

## Decode check

The P128 candidate generated 16 tokens at `33.19 tok/s`; the existing decode
path and allocation policy were unchanged. Long-context M11-A qualification
remains pending because this experiment changes prefill workspace only.

## Decision

**KEEP as an opt-in layer-major candidate.** The candidate is correct against
the current layer-major control at tested chunk boundaries and produces a
repeatable production-shaped gain, but P513 remains `45.02 tok/s`, far below
the `100 tok/s` gate. The default token-major path remains unchanged.

## Follow-up

The remaining gap requires a materially different production-shape quantized
GEMM/dataflow or a measured Amdahl ceiling. Larger logical chunks around the
existing B=4 microtiles were already neutral in EXP-0215.
