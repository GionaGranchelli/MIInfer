# EXP-0349 — M25 pinned Q6 FFN-down probe

**Status:** REJECTED for qualification; opt-in retained
**Milestone:** M25-L
**Date:** 2026-09-12
**Baseline:** `d170b31`
**Reference:** `mx-llama.cpp@2e9d29fe736969160f17476ec6f0a6298cee6966`

## Hypothesis

The current oracle trace showed the pinned Q6_K recurrent FFN-down projection
faster than MIInfer's Mx Q6_K projection at the exact B512 shape. Porting only
that contract may improve P512 without paying the rejected Q4 Gate/Up cost.

## Candidate

`MIINFER_MX_PINNED_FFN_DOWN=1` selects the already-ported pinned Mx Q6_K
repacked MMQ kernel only for recurrent wide-prefill Q6_K FFN-down projections.
Q4 Gate/Up, QKV, SSM-out, attention, and the qualified preset remain unchanged.
Q4 FFN-down layers, if present, continue using their existing path.

## Environment

- AMD Instinct MI50 / gfx906 / Wave64
- Qwen3.8-27B-Q4_K_M; SHA-256
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- SCLK/MCLK spot check: `1606/1000 MHz`
- clean `env -i`, fixed context capacity `1024`
- exact repeated-fox P512 prompt
- one-token continuation requested for the end-to-end screen

## Isolated B512 result

The existing `miinfer-m24-projection-bakeoff` harness measured layer 60
`blk.60.ffn_down.weight` (`5120 x 17408`, Q6_K):

| contract | Q6 MMQ us | max contract error |
| --- | ---: | ---: |
| current Mx staged | 8144.632 | `6.0e-7` |
| pinned Mx | 7286.712 | `6.0e-7` |

The pinned kernel was `10.53%` faster in isolation.

## End-to-end result

Three fresh interleaved process pairs used the full H/I vector and differed
only by `MIINFER_MX_PINNED_FFN_DOWN=1`:

| pair | control ms | pinned-Q6-down ms |
| ---: | ---: | ---: |
| 1 | 2477.94 | 2487.31 |
| 2 | 2498.38 | 2509.68 |
| 3 | 2453.67 | 2490.39 |
| **median** | **2477.94** | **2490.39** |

The candidate regressed by `12.45 ms` (`+0.502%`) at the P512 median, from
`206.623` to `205.590 tok/s`. All six processes completed, processed 512
prompt tokens, produced a finite first decode token, and reported the same
`21,993,242,964 B` setup allocation.

## Interpretation

The isolated pinned Q6 result is real, but it does not compose into a P512
win. The recurrent FFN-down differential is therefore too small or too
overlapped with the rest of the full execution to justify promotion. This
reinforces the M25-L conclusion: isolated kernel speed is not enough to
explain the remaining stretch gap.

## Decision

**REJECT** for the qualified preset. Keep the narrow selector as an opt-in
oracle comparison; do not add it to `m25_hi_qualified` or pursue another
FFN-down geometry variant without a new full-pipeline hypothesis.

## Follow-up

Return to the boundary-matched recurrent QKV/GDN execution differential. The
current measured stretch gap remains an execution-contract question, not a
reason to replace the qualified H/I path with the pinned FFN-down kernel.
