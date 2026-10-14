# EXP-0431 — Packed-U128 GQA6 unroll-8

**Status:** REJECTED — exact on both MI50s, 24–26% slower

## Hypothesis

Increase the packed-U128 GQA6 K/V microtile from four to eight positions. The
larger unroll should halve loop and LDS synchronization iterations while
retaining each query's exact arithmetic and online-softmax order.

## Method

- Model: Qwen3.8-27B Q4_K_M; Q8/Q8 cache.
- Restored prefix: 73,099 tokens; identical tokenized suffix: 1,024 tokens.
- HIP Graph replay; packed-U128 GQA6 unroll-4 control versus unroll-8.
- One sequential control/candidate pair per card; both cards ran concurrently.
  Compare timings only within each card.
- ROCm 7.14 complete image. Container device access used
  `--security-opt label=disable` after an isolated HIP copy probe required it.
- The graph trace reported all 16 attention layers on the requested route.
- Both processes exited 0. Thermal telemetry peaked at 64 C on GPU0 and 58 C
  on GPU2, below the 80 C warning threshold.

## Results

| MI50 / BDF | Control | Unroll-8 | Candidate/control time | Exact logits/state/tokens |
| --- | ---: | ---: | ---: | :---: |
| GPU0 / `0000:04:00.0` | 20.270 s | 25.505 s | 1.258x | yes |
| GPU2 / `0000:88:00.0` | 22.160 s | 27.465 s | 1.239x | yes |

Both candidates restored the 73,099-token prefix, replayed one 1,024-token
suffix tile, reached position 74,123, and matched full logits, recurrent-state
trace, generated tokens, and finite-value checks. The timing is one sample per
route per card; it is a focused screen, not a timing distribution. The
consistent slowdown is large enough to reject without repeating.

Raw artifacts are on Machinist:

- `/home/machinist/m31-gqa6-longctx-results/EXP-0431-gqa6-unroll8/sessionpair-gpu0.log`
- `/home/machinist/m31-gqa6-longctx-results/EXP-0431-gqa6-unroll8/sessionpair-gpu2.log`
- `/home/machinist/m31-gqa6-longctx-results/EXP-0431-gqa6-unroll8/telemetry-m31-gqa6-unroll8-gpu0-r2.log`
- `/home/machinist/m31-gqa6-longctx-results/EXP-0431-gqa6-unroll8/telemetry-m31-gqa6-unroll8-gpu2-r2.log`

## Decision

Revert unroll-8 and preserve packed-U128 GQA6 unroll-4. The screen does not
profile resource use, so it does not establish why the larger unroll is
slower. Do not continue with a larger unroll. The next change should alter the
cold-prefill execution schedule or attention work distribution, not this
microtile width.
