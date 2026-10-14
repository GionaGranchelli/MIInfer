# EXP-0430 — gfx906 FP16 dot2 QK for packed Q8 GQA6

**Status:** REJECTED — real-session logits/state differ and both cards are slower

## Hypothesis

Replace eight scalar FP32-by-Q8 QK products per lane with `v_dot2_f32_f16`,
using FP16 high and residual query terms. Keep Q8 K/V staging, GQA6 mapping,
Split-K boundaries, softmax order, and the production route as control.

## Method

- Source baseline: `46693864531accb226b1fccec0a3aef769371005` plus an
  uncommitted candidate patch (patch SHA-256
  `a351431a13326fdc1cca00cbb7f716cfd26d7b54d23723a76e95cb4088cafcf0`).
- Model: Qwen3.8-27B Q4_K_M; Q8/Q8 persistent session at 73,099 tokens;
  same tokenized 1,024-token suffix; HIP Graph replay; packed-U128 GQA6
  control versus dot2 candidate.
- One pair per card, run concurrently across GPU0 and GPU2; timings compared
  only within each card. Route traces confirmed all 16 attention layers used
  the requested control and candidate routes.
- ROCm 7.14 complete image; binary SHA-256
  `437556a4650b13835d62a8bfc98cc45fa4cdc36c553e68240aafaba061f1f2e4`.
- An isolated 4 KiB HIP copy failed with the default container label and
  passed on both cards when that container used
  `--security-opt label=disable`. This run uses that per-container option;
  the host SELinux policy was not changed.

## Results

| MI50 / BDF | Control | Dot2 | Candidate / control | Full logits / state | First and second token | Peak junction |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| GPU0 / `0000:04:00.0` | 20.276 s | 21.584 s | 1.064x time (0.939x speed) | different | equal / equal | 66 C |
| GPU2 / `0000:88:00.0` | 22.093 s | 23.572 s | 1.067x time (0.937x speed) | different | equal / equal | 57 C |

Each candidate completed 73,099-token restoration, one 1,024-token suffix,
one graph replay tile, finite logits/state, and reached position 74,123.
The full-model comparison reported maximum absolute logit difference
`1.232464790` and cosine `0.999299426`. Both cards reported exact-logit and
recurrent-state checks false, despite the same first two greedy tokens.
This is one pair per card, not a timing distribution; correctness alone
rejects the candidate, and both single samples also ran slower.

## Decision

Reject the FP16 high/residual dot2 QK route. Do not promote it, repeat its
timing, or change the production default. Restore the source to the exact
packed-U128 GQA6 baseline. Continue with an arithmetic-preserving Stage1
dataflow change; the existing paired-token path's roughly 1% gain and the
rejected tile4/tile8 and approximate-Q paths do not close the prefill gap.
