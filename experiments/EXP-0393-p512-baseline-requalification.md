# EXP-0393 — Requalify the aligned P512 baseline

**Status:** `BASELINE_NOT_REQUALIFIED`
**Disposition:** `LEARN`
**Date:** 2026-09-24
**Current commit:** `7aea7e3`

## Question

Can current HEAD once again be treated as a stable, qualified aligned-P512
baseline after EXP-0392 classified the earlier slow state as
`VARIANCE_NOT_REPRODUCED`?

No attribution or optimization was performed.

## Contract

- AMD Instinct MI50 / gfx906 / Wave64
- Qwen3.8-27B-Q4_K_M, `Q4_K_M`, SHA-256
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- HIP `7.1.52802-9999`, HIP Clang `20.0.0.rocm`
- `MIINFER_PRESET=m25_hi_qualified`, context capacity 1024
- exact repeated-fox P512 prompt, exactly 512 tokens
- `max_new_tokens=0`, fresh process per sample, one unrecorded warm-up
- no EXP-0376/0390/0391/0392 timing or profiler selectors
- mx sentinel: `llama-bench -p 512 -n 0 -r 1`, same model and pinned build
- six serialized mx/MIInfer pairs

The first six-sample attempt was not used for qualification because its
telemetry helper was invoked incorrectly. Its benchmark outputs are retained
at `/tmp/exp0393-p512/`; the admitted result below is the corrected
telemetry-backed set at `/tmp/exp0393-p512-telemetry/`.

## Results

### MIInfer P512 samples

| sample | total ms | prefill tok/s | processed tokens | allocation B |
|---:|---:|---:|---:|---:|
| 1 | 3279.57 | 156.23 | 512 | 21,993,243,028 |
| 2 | 2929.26 | 174.91 | 512 | 21,993,243,028 |
| 3 | 3242.53 | 158.02 | 512 | 21,993,243,028 |
| 4 | 3373.10 | 151.90 | 512 | 21,993,243,028 |
| 5 | 2469.72 | 207.74 | 512 | 21,993,243,028 |
| 6 | 2930.39 | 174.76 | 512 | 21,993,243,028 |

Median total latency is `2929.825 ms`; minimum `2469.72 ms`, maximum
`3373.10 ms`, range `903.38 ms`. Median throughput is approximately
`174.76 tok/s`.

The established healthy class is approximately 2.5 seconds. Four of six
samples are in the 2.93–3.37 second range; this is not a unimodal distribution
centered on the qualified class. The prior distinct 4.8–5.2 second state did
not recur, but the baseline still fails the current qualification gate.

### mx sentinels

| sample | latency ms | throughput tok/s |
|---:|---:|---:|
| 1 | 2308.819 | 221.758 |
| 2 | 2309.517 | 221.691 |
| 3 | 2312.017 | 221.452 |
| 4 | 2309.413 | 221.701 |
| 5 | 2311.359 | 221.515 |
| 6 | 2309.703 | 221.674 |

mx median is `2309.615 ms`; range is `3.198 ms` and it remained stable.

## Hardware and runtime audit

The corrected run recorded 1,699 samples at 250 ms intervals:

- SCLK was `(1606Mhz)` in every sample.
- MCLK was `(1000Mhz)` in every sample.
- junction temperature ranged from `35 C` to `60 C`.
- peak reported socket power was `190 W` against the `225 W` policy.
- peak reported VRAM use was `22,945,972,224 B`.
- no ROCm error or stale GPU process was observed; no overlapping benchmark
  process was present before the run.
- allocation bytes were constant at `21,993,243,028 B` in every MIInfer
  sample.
- `rocm-smi` did not expose an authoritative throttle field in this capture;
  clocks remained fixed and no throttle event was observed.

## Qualification decision

`BASELINE_NOT_REQUALIFIED`.

The six-sample distribution is materially outside the established ~2.5 second
regime and is not unimodal around it. mx, clocks, allocation, route contract,
model, preset, and token count were constant, so the result is not explained
by the qualification exclusions. The exact cause remains outside this
experiment's scope.

## Gate state

- B128: **DEFERRED**; not authorized by this experiment.
- B64: **REJECTED — CURRENT M28 FRONTIER**.
- B4: **BLOCKED**.
- recurrent/attention variance profiling: **DEFERRED**.
- no source bisect or optimization was authorized.

**ONE next PRIMARY:** baseline stability remains the PRIMARY; identify the
minimal reproducible MIInfer runtime boundary for the current ~3-second slow
state under a separately authorized experiment. Do not run B128 until P512
requalifies.

No new GPU math kernel or residual scheduler optimization was implemented.

## Provenance

- experiment commit: `e40af60`
- graph SHA-256: `efbcebead4114f583000da5c5ff482ffb61f80842d69048f5ff9a0023dc4d21f`
- graph refresh: `rtk graphify update .`
- working tree was clean before the experiment; graphify outputs are recorded
  in the following documentation refresh commit.
