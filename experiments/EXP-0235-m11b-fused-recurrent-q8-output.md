# EXP-0235 — M11-B fused recurrent Q8_1 output

## Hypothesis

The batched recurrent core already computes the gated FP32 output consumed by
the deferred Q5_K `ssm_out` projection. Emitting the canonical Q8_1 blocks in
that kernel should remove one quantization launch per recurrent layer chunk.

## Baseline

EXP-0234's causal recurrent-core B=4 path, with
`MIINFER_PREFILL_RECURRENT_Q8=0`. The existing per-token Q8_1 quantization
kernel remains between the recurrent core and the Q5_K `ssm_out` tail.

## Candidate

The B=4 recurrent core writes the same Q8_1 representation directly while it
has each gated output in registers. The old quantization launches remain the
fallback when this fusion is disabled or the batched `ssm_out` path is not
active. The control is explicit with `MIINFER_PREFILL_RECURRENT_Q8=0`.

## Environment

- GPU: AMD Instinct MI50, gfx906
- Model: `Qwen3.8-27B-Q4_K_M.gguf`
- Build: `mi50-release`
- ROCm: 6.4.0 / LLVM 20
- Prompt: repeated `hello` tokens
- `MIINFER_PREFILL_LAYER_MAJOR=1`
- recurrent-core B=4 enabled

## Correctness

P17 control and candidate both produced the same one-token `hello`
continuation. P513 control and candidate completed with finite output. Full
CTest passed `21/21`, including GPU correctness tests.

## Results

One matched production-shaped P513 pair:

| Path | Prefill | PP tok/s |
| --- | ---: | ---: |
| Q8_1 fusion disabled | 11177.62 ms | 45.90 |
| Q8_1 emitted by recurrent core | 11100.08 ms | 46.22 |

The candidate measured `+0.70%` for this pair. P17 remained within normal
run-to-run noise (`45.29` versus `45.55 tok/s`). This is a small local gain;
it does not move the projection Amdahl ceiling or close the 100 tok/s gate.

## Decision

**KEEP** as part of the opt-in layer-major recurrent B=4 path. Preserve the
explicit environment control for future A/B work.

## Follow-up

Pursue a whole-pipeline causal prefill design. Do not spend more time on
isolated Q8 launch removal unless profiling shows a larger share than measured
here.
