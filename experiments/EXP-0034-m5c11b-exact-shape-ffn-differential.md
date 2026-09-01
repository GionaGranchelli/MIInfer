# EXP-0034 — M5-C11b exact-shape FFN GEMV differential

**Status:** CLOSED — no FFN port selected; clock-matched rerun deferred  
**Milestone:** M5  
**Date:** 2026-09-01  
**Model:** Qwen3-8B Q4_0 (`Qwen3-8B-q4_0-b968826d.gguf`)

## 1. Question

Is the pinned gfx906 llama.cpp MMVQ primitive intrinsically faster than
MIInfer's production Q4_0 × Q8_1 GEMV for the exact Qwen3 Gate, Up, and Down
shapes, and is there a justified FFN kernel candidate for C11c?

## 2. Method and clock qualification

This was a measurement/forensics slice. No MIInfer production code or kernel
selection changed. The exact direct primitive comparison and raw artifacts are
retained in [`EXP-0008`](EXP-0008-direct-mmvq.md); the accepted K/V geometry
follow-up is retained in [`EXP-0009`](EXP-0009-kv-geometry.md).

The MI50 exposes stock DPM states through `/sys/class/drm/card2/device/`:

```text
SCLK: 925, 930, 1032, 1143, 1282, 1386, 1485, 1606, 1725 MHz
MCLK: 350, 800, 1000 MHz
```

The card was in `auto` performance mode. MIInfer telemetry sampled both the
925/930 MHz + 350 MHz low state and brief 1725 MHz + 1000 MHz active states,
then returned to the low state between bursts. A privileged `high` or
`profile_peak` control could not be run because the available `sudo` path
requires an interactive password. This is therefore not a clock-matched
MIInfer-versus-MMVQ claim.

The current-tree sequential Gate/Up sanity runs used 200 warmups and 10,000
HIP-event iterations:

| Shape | Median | Mean | Logical effective BW | Oracle |
|---|---:|---:|---:|---|
| Gate, 12288 × 4096 | 52.48 µs | 52.62 µs | 540.0 GB/s | PASS |
| Up, 12288 × 4096 | 52.80 µs | 52.59 µs | 536.8 GB/s | PASS |

A concurrent three-process sanity run was discarded as contaminated. The
current benchmark's legacy synthetic Down mode uses the 128-thread path, not
the selected production 256-thread Down path, so it was not used as a
production Down result.

## 3. Exact-shape direct comparison

The retained five-repetition, 20,000-sample matched-protocol direct results
are:

| Shape | MIInfer | MMVQ | MIInfer delta |
|---|---:|---:|---:|
| Gate, 12288 × 4096 | 60.320 µs | 61.600 µs | **−2.08%** |
| Up, 12288 × 4096 | 60.352 µs | 61.632 µs | **−2.08%** |
| Down, 4096 × 12288 | 57.600 µs | 64.928 µs | **−11.29%** |

The earlier seven-shape comparison also found MIInfer approximately equivalent
or faster for Q/O and the FFN shapes, while MMVQ was materially faster only
for K/V. EXP-0009 subsequently closed that K/V geometry gap: its accepted
128-thread path measured 11.52–11.84 µs against the 13.12 µs MMVQ result.

Both direct comparisons consume already-quantized Q8_1 activations and use
Q4_0 weights with FP32 accumulation. The historical primitive comparison
records an output-contract difference: MIInfer exposed FP16 while MMVQ exposed
FP32. This is a latency/structure comparison, not a claim of bit-identical
primitive outputs.

## 4. Kernel-structure findings

The pinned MMVQ source selects, for GCN batch-1 Q4_0, two Wave64 waves per
output row (`nwarps=2`, 128 threads) and `VDR_Q4_0_Q8_1_MMVQ=2`. Each thread
performs the Q4 nibble unpack and `v_dot4_i32_i8` algebraic zero-point
correction; one cross-wave partial reduction is stored in shared memory.

MIInfer's accepted production family is shape-specialized: Gate/Up use the
128-thread path, K/V use the accepted 128-thread geometry, and Down retains
the 256-thread reduction path. C8b and C8c showed that broadening or splitting
Down across additional Wave64 reductions regresses it materially. No new
mechanism-specific FFN candidate is identified by this evidence.

## 5. Decision

```text
C11b CLOSED — no FFN GEMV candidate selected
```

The exact-shape evidence does not justify porting MMVQ for Gate, Up, or Down:
MIInfer is already approximately tied or faster on those shapes under the
retained direct protocol, and the only historical MMVQ advantage (K/V) was
already addressed by EXP-0009. C11b therefore produces no C11c FFN-kernel
implementation.

This conclusion is qualified by the absence of a fixed-clock rerun. It is
strong enough to stop blind FFN geometry work, but not to claim that MIInfer
and llama.cpp have identical hardware operating points or end-to-end rates.

## 6. Follow-up

Obtain a privileged fixed-clock A/B if possible and rerun the production
micro/end-to-end controls before publishing a direct whole-runtime speed
ratio. If that is unavailable, move to a differential profile of non-FFN
families at the observed operating point. Do not port MMVQ or start another
Down geometry sweep without a new clock-controlled or hardware-counter-based
hypothesis.
