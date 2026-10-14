# EXP-0417 — GQA6 baseline versus cross-token Q8 fragment reuse

**Status:** REJECT — faster but fails full-model numerical parity
**Target:** Machinist MI50 GPU2, BDF `0000:88:00.0`, pinned ROCm 7.2.1 image

## Objective

Test whether the existing FP32-query/Q8-KV fragment-reuse schedule can improve
the verified production-arithmetic GQA6 path on a restored 73,099-token prefix.
The candidate reuses K/V tiles across 16 neighboring queries; GQA6 reuses K/V
across six query heads per token. This tests the current dominant Stage1
attention workload, not cold 64K prefill.

## Why this reopens prior query-tiled work

EXP-0405 rejected the earlier candidate for full-model parity and/or slower
exact execution against the then-current production Q8 route. GQA6 is now a
materially different, bit-identical baseline and accounts for 85.5% of the
profiled restored suffix tile. The prior query-tiled launcher also could not
consume dynamic persistent-session state during HIP Graph capture. This
experiment wires that state through and compares against GQA6 with the same 32
Split-K partitions. These changes are the new evidence; no query-tile sweep is
authorized by this record.

## Hypothesis and one variable

The 16-query fragment-reuse dataflow may reduce long-prefix KV traffic enough
to beat GQA6 Stage1. Compare the GQA6 route with the FP32-query/Q8-KV
fragment-reuse route, retaining the same model, restored session, tokenized
suffix, 32 splits, graph replay, compiler image and GPU. Only the attention
route changes. The launcher reads the graph's dynamic base position and token
count from `DevicePrefillState`.

## Gates

- **Dispatch:** emit `miinfer_q8_query_tiled_dispatch=1 graph_state=1` during
  graph capture; require the benchmark to report its graph replay tile.
- **Resource:** record VGPR, SGPR, spills, scratch and LDS for the selected
  gfx906 specialization; unexpected spills stop timing.
- **Correctness:** compare full logits (`max_abs <= 0.01`, cosine `>= 0.9999`),
  recurrent state trace, final hidden state and checked greedy tokens. Require
  finite outputs and successful restore/continuation.
- **Performance:** screen one matched 1,024-token restored suffix against the
  prior GQA6 result (~30.2 s). Keep only if correct and at least 10% faster;
  reject if it is not materially faster. Repeat only after passing.
- **Safety:** keep the 80 C warning and 85 C hard stop.

## Expected maximum impact and kill rule

Attention is 85.5% of this restored tile, so even an infinitely fast attention
route could improve this tile by at most about 6.9x. This does not establish a
cold-prefill speedup. Stop this route on a correctness failure, resource spill,
wrong graph dispatch, or no material end-to-end gain. If rejected, move to the
separately hypothesized packed Q8-to-LDS copy, which preserves GQA6 arithmetic.

## Result

The target rebuilt with the pinned image. The selected Q8 fragment-reuse
specialization reports 116 VGPR, 0 VGPR spills, 0 SGPR spills and no private
segment. The launcher uses 32 splits to match GQA6.

On the restored 73,099-token session at GPU2/BDF `0000:88:00.0`, the benchmark
reported a real graph replay tile and emitted
`miinfer_q8_query_tiled_dispatch=1 graph_state=1 token_count=1024` for all 16
attention layers. One matched pair produced:

| Route | Restore | 1,024-token suffix |
| --- | ---: | ---: |
| Production-arithmetic GQA6 | 2.408 s | 30.161 s |
| Q8 fragment reuse | 2.373 s | 22.853 s |

The candidate is 1.320x faster in this single screening sample, but fails the
full-model gate: final-logit maximum absolute error is `0.712` (quality
threshold `0.01`), the recurrent-state trace is not exact, and top two tokens
happen to match. Both outputs are finite and state position reaches 74,123.
Temperature guard passed and peaked at 60 C. **Reject**: equal generated tokens
do not excuse the logit/state divergence. Do not tune this fragment-reuse
schedule further without new numerical evidence.
