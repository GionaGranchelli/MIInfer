# EXP-0032 — M5-C10c FFN normalization-to-shared-Q8 fusion

**Status:** CLOSED — REJECTED for production selection
**Milestone:** M5
**Date:** 2026-09-01
**Control policy:** shared Gate/Up Q8 reuse, separate FFN normalization/Q8 stages
**Candidate:** opt-in `MIINFER_FFN_NORM_Q8_FUSION=fused`
**Model:** Qwen3-8B Q4_0 (`Qwen3-8B-q4_0-b968826d.gguf`)

## 1. Hypothesis

The FFN RMSNorm, norm-weight scaling, exact F32→F16 materialization, and
shared Gate/Up Q8_1 quantization could be performed in one gfx906 dispatch
without changing the bytes consumed by Gate and Up.

## 2. Candidate

The candidate uses one 256-thread workgroup. It performs the existing RMS
reduction, keeps normalized and norm-scaled values as distinct FP32
operations, applies `__float2half_rn`, and produces the existing exact Q8
block stream eight blocks at a time. The default path remains unchanged;
the candidate is selected only with `MIINFER_FFN_NORM_Q8_FUSION=fused`.

## 3. Environment and benchmark

The clean candidate build is commit `ce4e8f723b47`, Release, gfx906, with
HIP `7.1.52802-9999`, on the MI50. The observed clocks were approximately
930 MHz SCLK and 350 MHz MCLK, so these are balanced low-clock A/B results,
not canonical full-clock headline numbers. The model SHA-256 is recorded in
each JSON artifact.

The benchmark used prompt ID `14990`, eight warmup tokens, 64 measured decode
forwards, five iterations per run, and three strictly serial control/candidate
pairs with telemetry disabled. Results are retained under
`bench/results/20260901T-c10c-ab-clean/`.

## 4. Correctness

The clean real-model fused verifier ran at positions 1, 8, 16, 32, and 64. It
performed 180 FP16-materialization checks and 180 Q8 checks with zero
mismatches. The verifier artifact is
`bench/results/20260901T-c10c-fused-verify-clean/result.json`.

Release CTest passed 19/19. The control and candidate produced identical
64-token trajectories in all three valid A/B pairs, including the pinned
prefix `8,341,286,470,330,9707,11,330`.

## 5. Results

| Pair | Control tok/s | Fused tok/s | Control ms | Fused ms |
|---:|---:|---:|---:|---:|
| 1 | 54.133903 | 51.540588 | 1182.253578 | 1241.739803 |
| 2 | 54.094807 | 50.769606 | 1183.108029 | 1260.596735 |
| 3 | 54.149006 | 51.595812 | 1181.923811 | 1240.410754 |
| **mean** | **54.125905** | **51.302002** | **1182.428473** | **1247.582431** |

The fused candidate is **5.217% slower** in throughput, equivalent to a
**5.510% increase** in mean decode time. All runs were deterministic.

The clean P64 profiles show the intended structural reduction:

| Metric | Control | Fused |
|---|---:|---:|
| FFN norm/Q8 dispatches | 144 | 36 |
| Total dispatches | 1553 | 1445 |
| Synchronizations | 38 | 38 |
| Copy bytes | 589828 | 589828 |
| Allocations | 0 | 0 |
| FFN norm/Q8 GPU time | 1.423 ms | 2.182 ms |

The candidate removes 108 dispatches, but its single-workgroup reduction and
eight-at-a-time Q8 processing costs substantially more device time than the
existing parallel kernels. The dispatch reduction therefore does not produce
an end-to-end gain.

## 6. Decision

**REJECT for production selection.** The separate path remains the default.
The fused implementation and verifier remain available as an explicit
diagnostic candidate, but no tolerance or precision contract changed.

This result shows that one dispatch is not sufficient when the replacement
work decomposition reduces gfx906 parallelism. A future attempt would need a
different cross-workgroup design and a new measured hypothesis.

## 7. Artifacts

* Clean correctness/profile candidate: `bench/results/20260901T-c10c-fused-verify-clean/result.json`
* Clean control profile: `bench/results/20260901T-c10c-profile-control/result.json`
* Clean candidate profile: `bench/results/20260901T-c10c-profile-candidate/result.json`
* Serial A/B results: `bench/results/20260901T-c10c-ab-clean/`

