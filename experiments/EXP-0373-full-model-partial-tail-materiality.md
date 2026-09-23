# EXP-0373 — Refresh full-model partial-tail attribution

## Question

Does the aligned B128 partial-tail route materially change full-model behavior,
and how much prefill wall time does it recover?

## Current evidence and routes

The control was `m25_hi_qualified`. The candidate added only
`MIINFER_EXP0366_PARTIAL_TAIL=1`; all other selectors, model, quantization,
and context were unchanged. Existing EXP-0369 route evidence establishes that
complete B512 chunks are shared and only the final B128 route changes. Existing
same-route control/control and candidate/candidate deterministic checks remain
valid; no new route instrumentation was added.

## Full-model semantic comparison

Final-hidden and one-token logit dumps were compared from the existing P640
and P1664 state-oracle snapshots. The logit vector is the same ordinary decode
probe for each pair.

| Point | Hidden max / RMS / relative RMS / cosine | Logit max / RMS / relative RMS / cosine | Argmax | Top-10 overlap |
|---|---|---|---|---:|
| P640 | `2.00537 / 0.458363 / 0.0836302 / 0.996525` | `0.690930 / 0.127646 / 0.0323952 / 0.999489` | `561 = 561` | 9/10 |
| P1664 | `3.82567 / 0.508100 / 0.0741600 / 0.997332` | `0.510445 / 0.0713514 / 0.0238943 / 0.999715` | `33075 = 33075` | 10/10 |

Mean absolute hidden differences were `0.367183` and `0.392110`; logit mean
absolute differences were `0.100790` and `0.0558076` for P640 and P1664.
Final normalized state was not separately exported by the existing oracle.

Continuation was the decisive semantic check:

| Point | Control IDs | Candidate IDs | Result |
|---|---|---|---|
| P640 | `561,3841,13477,37550,33075,888,279,15217,5388,13,561,3841,13477,37550,33075,888` | identical | `CONTINUATION_IDENTICAL_16` |
| P1664 | `33075,888,279,15217,5388,13,561,3841,13477,37550,33075,888,279,15217,5388,13` | identical | `CONTINUATION_IDENTICAL_16` |

The result is semantically stable at the model boundary. The observed internal
drift is classified as `EXPECTED_WIDE_RECURRENT_NUMERICAL_DRIFT`, not material
model divergence.

## Recurrent state/history progression

P1664 state/history RMS by recurrent layer index is:

```text
layer:  0  1  2  4  5  6  8  9 10 12 13 14 16 17 18 20 21 22 24 25 26 28 29 30 32 33 34 36 37 38 40 41 42 44 45 46 48 49 50 52 53 54 56 57 58 60 61 62
state:  0  0  0  1.04e-5 1.04e-5 1.23e-5 1.88e-5 2.98e-5 1.80e-5 3.57e-5 2.23e-5 3.45e-5 1.70e-5 4.30e-5 4.12e-5 4.61e-5 5.58e-5 3.12e-5 5.48e-5 3.48e-5 8.49e-5 1.07e-4 1.05e-4 2.34e-4 1.98e-4 4.71e-4 4.58e-4 5.44e-4 9.64e-4 1.01e-3 8.11e-4 4.64e-4 6.81e-4 5.78e-4 5.99e-4 7.21e-4 5.12e-4 7.56e-4 6.12e-4 9.37e-4 1.90e-3 1.61e-3 1.76e-3 1.90e-3 1.81e-3 1.18e-3 7.04e-4 8.49e-4
history:0  0  0  3.52e-3 3.82e-3 5.41e-3 7.84e-3 1.11e-2 6.32e-3 1.54e-2 8.44e-3 1.15e-2 6.45e-3 1.56e-2 2.16e-2 2.04e-2 2.61e-2 1.40e-2 2.14e-2 2.18e-2 3.14e-2 6.09e-2 7.35e-2 9.94e-2 1.69e-1 1.95e-1 2.15e-1 3.51e-1 3.48e-1 3.69e-1 3.31e-1 3.16e-1 3.10e-1 3.30e-1 3.42e-1 3.28e-1 3.24e-1 3.61e-1 3.41e-1 4.03e-1 4.15e-1 4.26e-1 4.22e-1 4.28e-1 3.81e-1 2.85e-1 2.78e-1
```

The progression is accumulation with ordinary local variation, not a unique
catastrophic discontinuity. No L5→L6 layer forensics is authorized by this
experiment.

## Clean prefill timing

Fresh non-profiled runs were performed at the exact token counts. Peak tracked
allocation was `22,060,355,988 B` in every run.

| Point | Control ms / tok/s | Candidate ms / tok/s | Recovered ms | Recovered fraction |
|---|---:|---:|---:|---:|
| P640 | `15,811.06 / 40.48` | `3,497.01 / 183.01` | `12,314.05` | `77.89%` |
| P1664 | `21,375.60 / 77.85` | `8,508.29 / 195.57` | `12,867.31` | `60.20%` |

The clean runs were not profiled. Existing coarse route counters/profile
facilities were not enabled because their synchronization changes timing; the
route split is established by the selector and EXP-0369 tracing. No new
per-kernel attribution is claimed.

## Hardware

Telemetry after the matrix showed SCLK `1606 MHz`, MCLK `1000 MHz`, junction
`36 C`, and package power `28 W`. The external fan is physically fixed at full
speed; ROCm fan telemetry was not used. No throttle indication was observed in
the retained status output.

## Decision

**QUALIFY** — the aligned B128 route is semantically acceptable for the tested
P640/P1664 contracts and recovers material prefill wall time. Internal
wide-versus-scalar recurrent drift does not justify further L5→L6 archaeology.

## Next PRIMARY

Engineer and qualify arbitrary-remainder scheduling using the validated aligned
B128 route, minimizing residual scalar work. The next qualification should
cover remainder decomposition before any P4K/P8K curve expansion, followed by
fresh full-model attribution against the mx reference.

No new math kernel or production performance optimization was implemented.
