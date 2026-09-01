# EXP-0026 — M5-C8c Down long-K bottleneck attribution

**Status:** CLOSED — no production change
**Milestone:** M5
**Author:** MIInfer project
**Date:** 2026-09-01
**Baseline commit:** `cfecb2a966f9`
**Candidate:** existing `zero-point-128` two-Wave64 diagnostic path

## 1. Question

Why does the production Q4_0 × Q8_1 Down GEMV reach approximately 495 GB/s
while the Gate shape reaches approximately 564 GB/s, and is a split-K or
additional Wave64 decomposition justified?

## 2. Hypothesis

The Down shape (`M=4096, K=12288`) may be limited by its long reduction. A
two-Wave64-per-row decomposition could shorten the per-lane reduction and
improve latency if reduction serialization is the limiting mechanism.

## 3. Motivation and prior evidence

C8a measured the production-like controls at approximately 50.24 µs for Gate
and 57.28 µs for Down. C8b then tested four independent Wave64s per workgroup
with one output row per wave; it regressed Down to 83.36–84.80 µs. That result
rejected the simpler hypothesis that Down only needed more output-row
parallelism.

The repository already contains a 128-thread path with two Wave64s per row.
For Down, it is a direct split-K-style diagnostic: the two waves cover the
row's 384 Q4 blocks and reduce their partial sums in shared memory. Reusing it
keeps the experiment isolated and avoids adding another kernel variant.

## 4. Method

Environment and workload:

* AMD Instinct MI50 / gfx906 (`gfx906:sramecc+:xnack-`)
* Release build, HIP 7.1.52802-9999, Clang 20.0.0
* repository HEAD `cfecb2a966f9`, clean working tree
* Qwen3-shaped Down `4096 × 12288`, Q4_0 weights, Q8_1 activation
* 30 warmups and 1000 HIP-event iterations per run
* five control/diagnostic pairs, control then diagnostic
* the same deterministic input and streaming-rotate-3 weight regime as C8a

The control is the production `zero-point-dot` 256-thread kernel. The
diagnostic is `zero-point-128`, a 128-thread, two-Wave64-per-row kernel. Both
passed the existing quantized CPU oracle on every run.

The raw JSON measurements used for this record are retained locally under
`/tmp/mi50-c8c/`.

## 5. Results

| Path | Median latency | Effective logical bandwidth | Oracle |
|---|---:|---:|---|
| Production Down, 256 threads / 4 Wave64 per row | 57.280 µs | 494.65 GB/s | PASS |
| Two-Wave64 Down diagnostic | 73.120 µs | 387.49 GB/s | PASS |

The five production medians were identical at 57.280 µs. The five diagnostic
medians were identical at 73.120 µs. The two-Wave64 path is therefore 27.7%
slower than the production control. Its output remains correctness-valid, so
the result is a performance rejection rather than a semantic failure.

The refreshed Gate control measured 50.24–51.04 µs (557.01–564.11 GB/s) over
three runs and passed the same oracle. C8a's repeated Gate baseline remains
the canonical shape comparison; the refreshed run shows the same ordering.

## 6. Static gfx906 evidence

The installed ROCm toolchain could extract and disassemble the embedded
gfx906 code object. Hardware-counter profilers (`rocprof`, `rocprof-compute`,
and `omniperf`) are not installed in this environment, so this experiment
does not claim a hardware-counter classification such as HBM- or VALU-bound.

The production exact-metadata kernels have these static resources:

| Property | Gate, 128 threads | Down, 256 threads |
|---|---:|---:|
| Wavefront size | 64 | 64 |
| VGPRs | 45 | 45 |
| SGPRs | 16 | 17 |
| private segment / spills | 0 / 0 | 0 / 0 |
| dynamic LDS | 512 B | 1024 B |
| static barriers | 8 | 9 |
| static device instructions | 217 | 234 |
| static `v_dot4_i32_i8` | 8 | 8 |
| static global loads | 5 | 5 |
| static `s_waitcnt` | 23 | 26 |

The static data rules out a simple Down-only VGPR or spill problem: both
kernels use 45 VGPRs and report no spills. Down does have one additional
reduction barrier and a larger LDS tree. The loop executes over 384 Q4 blocks
per row instead of Gate's 128; with 256 threads, Down distributes about 1–2
blocks per thread, while the two-Wave64 diagnostic distributes 3 blocks per
lane. This is consistent with the measured rejection of the two-wave path,
but is not a complete dynamic bottleneck attribution.

## 7. Interpretation

The evidence does not support implementing split-K=2 or another arbitrary
Wave64 geometry candidate:

1. The existing two-Wave64-per-row path is a direct diagnostic and is 27.7%
   slower on Down.
2. C8b's four-independent-row Wave64 candidate is approximately 46% slower
   on Down.
3. Gate and Down have comparable static register use and no spills, so a
   straightforward register-pressure fix is not indicated.
4. The remaining tooling cannot provide dynamic memory-stall, occupancy, or
   issue counters on this host.

The current production four-Wave64-per-row decomposition is therefore the
best measured Down path. The measured gap is likely a shape-specific tradeoff
between long-K load/reduction work and the larger reduction tree, but C8c does
not distinguish memory transaction behavior from instruction/dependency
behavior. Further standalone Down geometry work is not justified without new
profiling capability or a concrete layout/prefetch hypothesis.

## 8. Correctness

All ten refreshed Down benchmark runs and three Gate runs passed the quantized
CPU oracle. Release
CTest passed all host and GPU tests: 8/8 host-only and 9/9 GPU-required,
19/19 total. No production kernel selection, precision policy, cache layout,
or model execution behavior changed.

## 9. Decision

```text
CLOSED — no split-K or geometry promotion
```

Keep the existing production Down kernel. Do not continue geometry roulette.
The direct split-K-style diagnostic is rejected on performance, and the
available evidence is insufficient to justify a new Down kernel.

## 10. Follow-up

Move the next optimization decision above the standalone Down GEMV: refresh
the end-to-end profile and investigate FFN activation/quantization
materialization or other higher-level reuse. Revisit Down only if a profiler
becomes available or a specific memory-layout/instruction hypothesis is
identified first.
