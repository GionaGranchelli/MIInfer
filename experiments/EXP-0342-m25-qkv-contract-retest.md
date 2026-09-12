# EXP-0342 — M25 QKV contract retest

**Status:** RETEST; opt-in only, not qualified
**Milestone:** M25 stretch follow-up
**Date:** 2026-09-12
**MIInfer source:** working tree after `3f6e412`
**Reference:** `mx-llama.cpp@2e9d29fe736969160f17476ec6f0a6298cee6966`

## Hypothesis

The pinned oracle's large-batch Q6 MMQ schedule is faster on the recurrent
QKV shape even though the same contract was rejected when applied globally or
to recurrent FFN.

## Candidate

Add `MIINFER_MX_PINNED_QKV=1`. When Mx prefill is active, it selects the pinned
large-batch Q6 MMQ kernel only for recurrent QKV. The qualified preset sets
the selector explicitly to `0`; no FFN, attention-O, decode, or default path
changes. The M24 projection bakeoff now accepts `qkv` so this exact shape can
be measured without changing the runtime preset.

## Environment

- AMD Instinct MI50 / gfx906 / Wave64
- Qwen3.8-27B-Q4_K_M; SHA-256
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- recurrent layer 60, Q6_K `attn_qkv.weight`
- exact shape: rows `10240`, columns `5120`, batch `512`
- clean `env -i` for isolated measurements
- qualified H/I vector for end-to-end measurements

## Isolated result

`miinfer-m24-projection-bakeoff MODEL qkv q6 60` uses the same Mx activation
quantizer and packed Q6 weights as production. The internal seven-sample
median was:

| path | Q6 MMQ us | Q8 + MMQ us |
|---|---:|---:|
| H/I control | `4516.955` | `4581.914` |
| pinned QKV | `4032.155` | `4096.795` |
| delta | `-10.72%` | `-10.59%` |

The contract check reported maximum absolute error `3.6e-7` against the host
Mx reference at sampled rows/tokens. No non-finite output occurred.

## End-to-end screen

Three fresh-process pairs were run serially on the same GPU. The source prompt
was the exact 512-token repeated-fox prompt used by the H/I screen.

```text
control:    2486.07, 2500.70, 2761.63 ms  (median 2500.70 ms)
pinned QKV: 2455.94, 2516.61, 2492.25 ms  (median 2492.25 ms)
```

The median difference is `-8.45 ms`, or `-0.34%` (`205.44` versus `204.74
tok/s`). The `2761.63 ms` control is retained as a raw outlier; it was not
silently removed. The screen did not meet the six-sample promotion protocol,
and the result is below the observed run-to-run spread.

## Negative GDN retest

Because the oracle trace also showed a faster GDN event, the existing Mx GDN
standalone check was changed to the production B512 length. Explicitly
unrolling the fixed state/column loops changed the median from `2834.876 us` to
`2832.636 us` (`0.08%`) with unchanged output/state error (`<1.5e-7`). The
oracle's 32-bit shuffle mask cannot be copied literally: current HIP headers
require an eight-byte mask and reject the 32-bit form at compile time. The
unroll candidate was removed; the B512 benchmark length remains because it is
the relevant production check.

## Decision

**RETEST / KEEP OPT-IN.** The pinned QKV contract is a real isolated win and
is retained behind `MIINFER_MX_PINNED_QKV=1`, but it is not part of
`m25_hi_qualified` and does not close the stretch gap. The next work should
measure a like-for-like QKV/GDN execution contract, including materialization
and overlap, before promoting this selector or writing another kernel.

## Follow-up

1. Add a six-sample interleaved qualification only if the device remains
   stable and the selector is still justified by the contract differential.
2. Capture matching outer QKV/GDN spans in both runtimes; do not infer the
   end-to-end delta from isolated kernel names alone.
