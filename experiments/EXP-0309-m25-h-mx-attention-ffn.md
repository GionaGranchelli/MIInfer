# EXP-0309 — M25-H Mx attention FFN projections

## Hypothesis

The untouched attention-layer FFN projections are a measured P512 bottleneck.
Reusing the already-qualified Mx repacked MMQ path for attention Gate, Up, and
Down should reduce the attention deferred tail without changing recurrent
state, KV handling, or decode.

## Baseline

The control was the committed resident M23 attention path at baseline commit
`a6f7bb1`, with the established Mx recurrent projections and Mx GDN scan
enabled. Attention Gate/Up used Q4 M23 repacked MMQ and Down used Q6 M23
repacked MMQ.

## Candidate

`MIINFER_PREFILL_WIDE_MX_REPACKED_ATTN_FFN=1` packs attention FFN weights in
the existing Mx Q4/Q6 repacked layout, reuses the Mx Q8_1 activation scratch,
and dispatches the existing Mx MMQ kernels. The path is opt-in and applies
only to wide attention prefill; scalar decode remains M23.

## Environment

- AMD Instinct MI50 / gfx906 / Wave64
- Qwen3.8-27B-Q4_K_M; SHA-256
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- SCLK/MCLK `1606/1000 MHz`
- context capacity 1024; exact 512-token repeated fox prompt
- established row-128 resident-all M23/Mx profile and `--max-tokens 0`
- pinned Mx reference: `/home/fedora-workstation/Development/mx-llama.cpp`,
  commit `2e9d29fe736969160f17476ec6f0a6298cee6966`

## Benchmark

The standalone attention layer bakeoff used layer 3 at B512. The full-model
qualification used interleaved A/B process runs with a concurrent 250 ms
`rocm-smi --json` sampler.

## Correctness

- Layer output remained finite; max absolute error versus the scalar oracle was
  `0.555`, RMSE `0.009`.
- `qwen35-conv-batch-gpu` Release CTest passed.
- The exact P512 candidate completed successfully and its one-token
  continuation was `brown` (token ID `13477` in the established control).
- Candidate allocation was `18,188,222,804 B` versus control
  `19,108,282,708 B`.

## Results

Attention layer 3, B512:

| path | total GPU us | FFN gate/up ms | FFN down ms |
| --- | ---: | ---: | ---: |
| M23 control | 63331.318 | 28.186 | 14.877 |
| Mx FFN candidate | 42354.038 | 13.724 | 8.357 |

Clock-qualified full-model P512 raw latency samples were:

| path | latency ms | tok/s |
| --- | --- | --- |
| Mx FFN candidate | 2565.29, 2720.06, 2515.27, 2513.55, 2571.43 | 199.59, 188.23, 203.56, 203.70, 199.11 |
| M23 control | 2877.22, 2953.99, 2859.59, 2887.89, 2905.21 | 177.95, 173.33, 179.05, 177.29, 176.24 |

Median latency was `2565.29 ms` / `199.59 tok/s` for the candidate versus
`2887.89 ms` / `177.29 tok/s` for control, a `1.126x` improvement. All ten
telemetry captures observed only `1606/1000 MHz`.

## Interpretation

The measured attention FFN opportunity is real and carries through the full
P512 path. It nearly reaches the 200 tok/s gate by itself; the follow-up Mx O
projection experiment is recorded separately as EXP-0310.

## Decision

**KEEP as an opt-in path.**

## Follow-up

Test the measured M23 attention O projection with the same Mx repacked kernel
family. Keep the current M23 resident O copy for decode compatibility.
