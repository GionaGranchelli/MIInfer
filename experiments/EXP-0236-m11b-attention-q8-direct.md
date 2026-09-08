# EXP-0236 — M11-B direct attention Q8_1 emission

## Hypothesis

The tiled attention stage already has an optional Q8_1 output path. Using it
for deferred layer-major attention should avoid copying the 6,144-float gated
attention result and then launching a separate Q8_1 quantizer before the
batched output projection.

## Baseline

EXP-0234/0235 layer-major prefill. Each full-attention layer writes gated
attention to chunk storage, then `finish_prefill_batch` quantizes each of the
four rows before the batched O projection.

## Candidate

The existing tiled-attention reduction writes Q8_1 blocks directly into the
per-token chunk workspace during the position-ordered attention step. The
deferred tail consumes those blocks directly. Decode behavior is unchanged.

## Environment

- GPU: AMD Instinct MI50, gfx906
- Model: `Qwen3.8-27B-Q4_K_M.gguf`
- Build: `mi50-release`
- ROCm: 6.4.0 / LLVM 20
- Prompt: repeated `hello` tokens
- `MIINFER_PREFILL_LAYER_MAJOR=1`
- recurrent B=4 and recurrent Q8 fusion enabled
- Control: `MIINFER_PREFILL_ATTENTION_Q8=0`

## Correctness

Candidate and control produced the same P17 one-token continuation (`你好`).
The candidate completed P513 generation with finite output. The candidate
build succeeded. The committed 21/21 CTest result remains valid for the
unchanged production tree; this candidate was not retained for that suite.

## Results

One matched P513 pair (514 prompt tokens as reported by the tokenizer):

| Path | Prefill | PP tok/s |
| --- | ---: | ---: |
| Direct attention Q8_1 disabled | 11137.21 ms | 46.15 |
| Direct attention Q8_1 enabled | 11132.42 ms | 46.17 |

The observed difference was `+0.04%`, below useful measurement confidence.
The P17 pair was also noisy (`427.38 ms` control versus `405.42 ms`
candidate).

## Decision

**REJECT.** The existing optional kernel capability does not produce a
measurable end-to-end gain in the production-shaped path. The candidate code
was removed; retain the result as negative evidence.

## Follow-up

Do not revisit direct Q8 emission without a profile showing that the attention
copy and quantization pair is a material fraction of prefill time. The next
candidate must change the causal chunk's projection parallelism or dependency
structure.
