# EXP-0419 — Double GQA6 Q8 Stage1 key tile from U4 to U8

**Status:** REJECTED — exact, but 28–30% slower in first matched pair
**Baseline:** packed Q8 GQA6 U4 (`d930ed1`)
**Target:** Qwen3.8-27B Q4_K_M, restored 73,099-token state, 1,024-token suffix

## Objective and bottleneck

Reduce the measured Stage1 time for the exact GQA6 attention path. EXP-0412
measured Stage1 at 25.759 s of a 30.165 s restored suffix tile (85.5%). The
packed U4 LDS copy now improves whole-tile time by 1.361x but still performs
two CTA barriers for each four-key group.

## Hypothesis and one variable

An eight-key (`U8`) K/V tile halves the loop and CTA-barrier count relative to
U4 while preserving the same production arithmetic and key order. Compare
packed U4 against packed U8; vary only the number of keys processed per
stage/compute iteration. Query, softmax and value updates remain in increasing
key order. U4 remains the default and fallback.

## Gates

- **Resource:** pinned gfx906 build; require zero register spills and no
  unexpected scratch. Inspect VGPR/SGPR/LDS and emitted packed copy ISA before
  GPU timing.
- **Correctness:** restored graph replay; require exact final logits, recurrent
  state trace, generated tokens, finite values, and final position.
- **Performance:** screen one interleaved U4/U8 pair on each tested MI50. Keep
  only for at least 10% lower suffix time on both cards, with no correctness
  or resource failure; repeat only if the screen is promising.
- **Safety:** use the existing 80 C warning / 85 C stop guard.

## Compiler resource pre-gate

The pinned ROCm 7.2.1 compiler resource report gives packed U8 97 VGPR,
52 SGPR, zero spills, and zero scratch; packed U4 control gives 71 VGPR,
46 SGPR, zero spills, and zero scratch. Compiler-estimated occupancy drops
from 3 to 2 waves per SIMD for U8. Dynamic LDS requested by the launcher is
4,128 bytes for U8 versus 2,064 bytes for U4. This passes the no-spill gate,
but occupancy is a measured risk.

## Run setup notes

An initial dual-card launch was invalid: it omitted HIP device-node pass-through,
so both containers reported no usable HIP device. A second launch also omitted
`--tokens 1024`, which selected the harness default of 8,192 suffix tokens.
Those containers were stopped before any result was accepted (exit 137). The
valid run explicitly passes `/dev/kfd` and `/dev/dri` nodes and requests
`--tokens 1024` on both cards.

## Kill rule and end-to-end limit

Reject immediately on spills or any output/state mismatch. Reject if the
restored 1,024-token suffix median does not improve by at least 10%. Even a
large suffix win affects only the measured GQA6 Stage1 portion of cold
prefill; do not infer cold-64K time from this suffix experiment. If kept, run
one guarded cold-prefill measurement and refresh the 64K stage attribution.

## Result

The valid matched pair used the pinned ROCm 7.2.1 image, the same restored
73,099-token session, the same tokenized 1,024-token suffix, HIP Graph replay,
and one pinned MI50 per process. Both candidates dispatched packed U8 in all
16 attention layers. Outputs were exact on both cards: full logits, recurrent
state trace, generated tokens and finite-value checks all passed.

| GPU / BDF | Packed U4 | Packed U8 | U8 change | Exact |
| --- | ---: | ---: | ---: | :---: |
| GPU0 `0000:04:00.0` | 20.561 s | 26.760 s | +30.2% time | yes |
| GPU2 `0000:88:00.0` | 22.444 s | 28.825 s | +28.4% time | yes |

Rejected after this screen; the other two planned pairs were stopped to avoid
spending more GPU time on a candidate that regressed substantially on both
cards. Both stopped containers required SIGKILL after the 5-second SIGTERM
grace period (exit 137); each had completed the first exact pair before the
stop. Compiler resource data showed U8 at 97 VGPR and two waves per SIMD,
versus U4 at 71 VGPR and three waves per SIMD. This is consistent with the
regression but does not prove occupancy alone caused it. Artifacts are on
Machinist under `EXP-0419-U8-GPU0-20261014T1753Z` and
`EXP-0419-U8-GPU2-20261014T1753Z`.

No U8 implementation changes are retained. Continue with the proven packed U4
path and measure its direct cold-prefill impact.
