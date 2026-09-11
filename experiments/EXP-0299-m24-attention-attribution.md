# EXP-0299 — M24 full-attention attribution

## Hypothesis

The remaining P512 floor is not explained by attention projections alone. A
layer-level measurement of the first full-attention block should identify the
largest removable execution phases before another kernel experiment is chosen.

## Baseline and environment

- AMD gfx906 / MI50, Qwen3.8-27B-Q4_K_M
- ROCm 7.1.52802, Release build
- context capacity 1024, row-128 resident-all MMQ
- `MIINFER_PREFILL_LAYER_MAJOR=1`
- `MIINFER_PREFILL_WIDE_CHUNK=1`
- `MIINFER_PREFILL_FULL_LAYER_MAJOR=1` for the full-model control only
- `MIINFER_PREFILL_CHUNK=512`
- `MIINFER_PREFILL_WIDE_ATTN=1`
- `MIINFER_PREFILL_WIDE_REPACKED_MMQ=1`
- `MIINFER_M23_REPACKED_ROW128=1`
- `MIINFER_PREFILL_REPACKED_RESIDENT_ALL=1`
- exact 512-token repeated fox prompt, `--max-tokens 0`

## D0 — full-model reproduction

The complete resident-all command reproduced the qualified baseline:

| run | P512 latency | throughput | allocation |
|---|---:|---:|---:|
| resident-all control | 4838.43 ms | 105.82 tok/s | 22,801,772,884 B |

A deliberate control check with `MIINFER_PREFILL_REPACKED_RESIDENT_ALL` omitted
used the nonresident path and measured 12,597.15 ms / 40.64 tok/s at
19,394,924,884 B. This explains the earlier minutes-long control observation:
the switch set was incomplete, not evidence of a production regression.

## D1/D2 — layer-3 bakeoff

`miinfer-m24-attention-layer-bakeoff` executes the actual resident row-128
MMQ path for `FullAttentionLayer(model, 3)`, with one warmup and one timed
repeat. Phase timings are `prepare` (normalization, quantization, QK/V
projection), `attention` (Q postprocess, KV write, causal attention), and
`post_attention_ffn` (O projection, residual/norm, FFN and residual). Internal
stage events are also emitted. The harness also runs the same input sequence
through the position-ordered scalar resident-MMQ path and reports output parity
against that control. Values below are GPU milliseconds.

| batch | whole layer | prepare | attention | post-attention + FFN | tracked layer bytes |
|---:|---:|---:|---:|---:|---:|
| 128 | 18.480 | 4.032 | 0.357 | 14.075 | 375,369,616 |
| 256 | 32.865 | 6.417 | 0.866 | 25.566 | 437,563,280 |
| 512 | 59.359 | 11.106 | 2.621 | 45.616 | 561,950,608 |

Scalar parity was finite at every batch. Maximum absolute differences were
`0.012` (B128), `0.026` (B256), and `0.033` (B512), with RMSE `0.001` in
each case. These are scheduling-control results, not independent external
reference validation. The reported allocation remains the timed wide layer;
the temporary scalar control is not included.

B512 internal stages were:

| stage | GPU ms |
|---|---:|
| QK projection | 9.696 |
| query norm/RoPE | 0.145 |
| K norm/RoPE/KV store | 0.071 |
| V projection | 1.289 |
| causal attention | 2.389 |
| O projection | 4.812 |
| residual/post-norm | 0.092 |
| FFN Gate/Up | 26.479 |
| SwiGLU | 0.151 |
| FFN Down | 13.927 |
| residual | 0.050 |

The combined QK projection and the FFN stages dominate this layer. The causal
attention core is small at B512, so an attention-only projection change cannot
be assumed to move the end-to-end target materially.

## Correctness

The existing focused suite passed all 24 tests, including GPU primitives,
cached attention determinism, forward, decode and decode-sequence checks.
The scalar parity gate is finite and bounded as reported above. The new
harness still does not claim independent canonical tensor parity; the M6-A4
external fixture remains the reference-output gate before integrating any
candidate path.

## Decision

**KEEP as M24-D attribution infrastructure.** Do not optimize Q/K/V/O in
isolation yet. The next candidate should target the largest measured
post-attention/FFN work or reduce the unprofiled prepare/materialization cost,
with absolute milliseconds saved propagated to the 4,838 ms P512 baseline.

## Follow-up

1. Add a canonical layer-output comparison to the layer harness.
2. Split full-model `deferred_prefill_tail` ownership into recurrent and
   attention counters (the report now emits both).
3. Only then run one-projection FP16 A/B tests for QK, V and O.
