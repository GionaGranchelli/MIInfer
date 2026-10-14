# EXP-0428 — Multi-plane Q8 query dot4 screen

**Status:** REJECTED — FULL-MODEL ACCURACY GATE FAILED

## Claim under test

Four Q8 query planes (base plus three residual planes) may recover the
full-model accuracy lost by single-plane and one-residual dot4 while keeping
at least a 1.20x cold-prefill speedup over production Q8. This is a screening
test only; it does not qualify the current packed-U128 GQA6 route or 64K.

## Why this is the next screen

The exact packed-U128 GQA6 path is the current control. Its 4K restored suffix
comparison and late-tile profile already exist in EXP-0412. EXP-0427 rejected
larger query CTAs. Separately, the single-plane and one-residual dot4 paths
failed the full-logit gate. The existing three- and four-plane query
quantizers are the only unmeasured accuracy-recovery variants in that family.

## Method

- Machinist MI50s, one fixed executable, model, 8,192-token prompt, ROCm 7.2.1
  image and runtime settings.
- Run exact Q8 control followed by the four-plane candidate on each card.
  GPU0 and GPU2 run concurrently; compare only within each card.
- Candidate enables `MIINFER_M31_Q8_DOT4_ATTENTION=1` and all three query
  residual planes. All GQA6 candidate selectors remain disabled for this
  screen, so it tests the numerical method in its existing dot4 route.
- Save full final logits and first two greedy tokens. Require finite outputs,
  max absolute logit error <= 0.01, cosine >= 0.9999, equal first two tokens,
  and at least 1.20x speedup on both cards to continue.
- Keep the 80 C warning / 85 C stop guard. Do not run 64K unless this screen
  passes; a later candidate must combine with GQA6 before it can replace the
  current best-known route.

## Results

The first launch under the frozen ROCm 7.2.1 image failed during the Q8
control on both cards with ROCr exit 134 and `Memory critical error ... Reason:
Memory in use`; no candidate ran. A 128-token full-model smoke on each card
then completed under the locally installed `docker.io/mixa3607/rocm-gfx906:7.14-complete`
image. The 8K A/B was rerun there. This changes the runtime from the pinned
campaign environment, so these results are valid only as an internally matched
ROCm 7.14 screen and are not merged with ROCm 7.2.1 timings.

The matched screen completed on both MI50s with the same candidate binary,
model, prompt, and capacity. Each card ran production Q8 control followed by
four-plane dot4. Both processed 8,192 prompt tokens at capacity 65,536, emitted
248,320 finite logits, and returned exit code 0. The first two greedy tokens
matched on both cards (11, 443). Full-logit comparison failed the declared
gate on both cards:

| MI50 | BDF | Q8 control | Four-plane dot4 | Speedup | Max absolute logit error | Cosine |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GPU0 | `0000:04:00.0` | 54.842 s | 39.913 s | 1.374x | 1.09448 | 0.9989098 |
| GPU2 | `0000:88:00.0` | 56.240 s | 42.953 s | 1.309x | 1.09448 | 0.9989098 |

The measured peak junction temperatures were 69 C on GPU0 and 62 C on GPU2;
neither thermal guard fired. The screen has one control/candidate pair per
card, so its timings are not repeat-qualified. The candidate's speedup cannot
be used because the full-logit error exceeds the `max_abs <= 0.01` and
`cosine >= 0.9999` thresholds by a large margin. Reject four-plane dot4 and do
not proceed to 64K with this quantization strategy.

Artifacts are under
`/home/machinist/m31-gqa6-longctx-results/EXP-0428-q8-dot4-multiplane-rocm714-20261015T0010Z/`.
The candidate executable SHA-256 is
`427ee6d108418380b03a9530d013087de8ee0f9f3d2aef24abc18b6b54caa76a`; the
model SHA-256 is
`7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`. The
ROCm 7.14 image ID was `b8b8200afce88ce8d17bac793c37c50425a339b5408027d1ed8f19803a104173`.
The original 7.2.1 control failures and this 7.14 recovery identify the runtime
image as a relevant variable, but do not by themselves prove the underlying
cause of the ROCr failure.
