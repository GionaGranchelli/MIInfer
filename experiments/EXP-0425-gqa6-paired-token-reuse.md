# EXP-0425 — Paired-token GQA6 K/V reuse

**Result:** exact and repeatably faster on a restored 73K-prefix 4K suffix;
retain as opt-in. It does not establish a cold-prefill improvement.

## Change

The candidate CTA handles two neighboring query tokens and stages their
shared Q8 K/V history once. It retains separate query/head softmax state,
causal bounds, Split-K ranges, and ordered arithmetic updates. The dynamic
prefill-state selector is included in the HIP Graph cache identity so control
and candidate graphs are captured separately.

The pinned ROCm 7.2.1 build succeeded. The gfx906 candidate specialization
uses 83 VGPRs with no VGPR/SGPR spills or private-segment bytes. Binary
SHA-256: `228eb8cc00d143f5c394859467ea6476fc0f5e93567376787ee332b70afad41e`.

## Method

- Source baseline: `99142969c49445b71fe15e68f0ff574585900cbb` plus this
  uncommitted candidate.
- Model: Qwen3.8-27B Q4_K_M; pinned image
  `localhost/miinfer-m31-rocm721-libdw@sha256:422b5d15d384d78cab724a0190ef739d31212674ebc1ae8f0f6cf538b1f5ea8f`.
- Same restored Q8/Q8 checkpoint at 73,099 tokens and same tokenizer-backed
  4,096-token suffix; four 1,024-token graph tiles per path.
- Three control/candidate pairs per card, sequential in the same process.
  Control is packed-U128 GQA6; candidate changes only to paired-token GQA6.
- Restore time is excluded from suffix timing and recorded separately.
- Both cards ran concurrently; timings are compared within each card and not
  pooled across cards.

## Results

| MI50 / BDF | Control median (range) | Candidate median (range) | Paired speedup | Correctness |
| --- | ---: | ---: | ---: | --- |
| GPU0 / `0000:04:00.0` | 83.272 s (83.260–83.833) | 82.724 s (82.622–82.787) | 1.007x | exact logits, state, tokens; finite |
| GPU2 / `0000:88:00.0` | 91.032 s (91.025–91.047) | 90.104 s (90.078–90.173) | 1.010x | exact logits, state, tokens; finite |

All six pairs restored 73,099 tokens and passed full-logit equality,
recurrent-state-trace equality, greedy-token equality, and finite-value
checks. Dispatch traces showed the intended graph-state route in all 16
attention layers. Restore took 2.4–2.5 seconds per run. The timing ranges do
not overlap within either card, but the improvement is small: about 0.55 s on
GPU0 and 0.93 s on GPU2 for the 4K suffix.

Thermal guards sampled 390 times per card. Peak junction temperature was
77 C on GPU0 and 66 C on GPU2, below the 80 C warning threshold. Both jobs
exited successfully and their containers were stopped after completion.

Raw evidence on Machinist:

- `/home/machinist/m31-gqa6-longctx-results/EXP-0425-pair-token-4k/gpu0/`
- `/home/machinist/m31-gqa6-longctx-results/EXP-0425-pair-token-4k/gpu2/`

## Decision and limits

Keep the route opt-in as a small additive exact gain. Do not change the
production default or claim a cold 64K improvement. The existing late-context
profile attributes about 80% of a 1K tile to attention Stage1, but this run did
not profile the candidate's internal stages; it only establishes the
end-to-end suffix effect. At roughly 1%, this change alone cannot close the
current cold-prefill gap. The next optimization must reduce more Stage1 work
per CTA, not spend more time tuning the paired-token candidate.
