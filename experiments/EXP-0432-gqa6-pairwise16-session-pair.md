# EXP-0432 — Packed-U128 GQA6 pairwise16 reduction

**Status:** REJECTED — logits, recurrent trace, and generated tokens differ

## Hypothesis

Combine the offset-16 lane pair locally and use a 16-lane reduction to reduce
cross-lane shuffle work while retaining Q8 products and the GQA6 KV-sharing
route.

## Method

- Model: Qwen3.8-27B Q4_K_M; restored Q8/Q8 session prefix at 73,099 tokens.
- Identical tokenized 1,024-token suffix, with HIP Graph replay.
- Packed-U128 GQA6 control versus opt-in pairwise16 candidate.
- One ordered control/candidate pair on each MI50; cards ran concurrently.
- ROCm 7.14 complete image. Container used
  `--security-opt label=disable`; the host SELinux policy was unchanged.
- Both reached position 74,123 and reported one graph replay tile.

## Results

| MI50 / BDF | Control | Pairwise16 | Candidate/control time | Exact logits/state/tokens | Peak junction |
| --- | ---: | ---: | ---: | :---: | ---: |
| GPU0 / `0000:04:00.0` | 20.400 s | 19.550 s | 0.958x | no | 63 C |
| GPU2 / `0000:88:00.0` | 22.286 s | 21.234 s | 0.953x | no | 62 C |

Each process completed the restore and suffix work with finite values. Both
reported `session_logits_exact=0`, `recurrent_state_trace_exact=0`, and
`generated_tokens_equal=0`; first tokens also differed. The 4–5% lower time is
one observation per route per card, in fixed control-then-candidate order. It
is not a reliable speed estimate and cannot justify adoption. Route labels
were emitted by the benchmark harness; per-layer dispatch tracing was not
enabled in this run.

## Decision

Reject pairwise16 and restore the packed-U128 GQA6 source unchanged. The
reduction mapping failed the exact numerical contract. Do not pursue this
shuffle reduction further without a materially different exact formulation.

Next, run the required original-Q8 versus GQA6 continuation comparison at
1,024 and 4,096 suffix tokens from the compatible 73,099-token checkpoint,
then collect a low-overhead late-context stage profile.

Raw artifacts on Machinist:

- `/home/machinist/m31-gqa6-longctx-results/EXP-0432-gqa6-pairwise16/gpu0/run.log`
- `/home/machinist/m31-gqa6-longctx-results/EXP-0432-gqa6-pairwise16/gpu2/run.log`
- `/home/machinist/m31-gqa6-longctx-results/EXP-0432-gqa6-pairwise16/gpu0/telemetry.log`
- `/home/machinist/m31-gqa6-longctx-results/EXP-0432-gqa6-pairwise16/gpu2/telemetry.log`
