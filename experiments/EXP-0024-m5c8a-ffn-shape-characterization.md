# EXP-0024 — M5-C8a FFN projection shape characterization

**Status:** COMPLETE — baseline recorded; no production change  
**Milestone:** M5  
**Date:** 2026-09-01  
**Baseline:** `b1d43d9fbf74`

## Hypothesis

The FFN projection family is the largest named GPU category in the M5-C7
profile. Before changing the common Q4_0 × Q8_1 GEMV primitive, measure the
three real Qwen3 FFN shapes independently and determine whether Gate/Up or
Down is the more promising geometry target.

## Method

The existing `miinfer-q4-q8-gemv-bench` was used in kernel-only `gemv` mode.
It generates deterministic FP16 inputs and weights, quantizes them to the
canonical Q8_1/Q4_0 layouts, checks the result against the Q4_0 × Q8_1 CPU
oracle, and times only the selected GEMV launch. Each recorded shape used 30
warmups and 300 measured iterations; three independent runs were collected
for the current production-like geometry.

This is a primitive characterization, not a full-model result. The benchmark
uses Q8_1 metadata, while the current Qwen3 production path uses its validated
exact-Q8 metadata variant. No runtime kernel selection or precision policy was
changed by this experiment.

## Environment and workload

* Model-shaped dimensions: Gate/Up `M=12288, K=4096`; Down `M=4096, K=12288`
* GPU: AMD Instinct MI50 / gfx906 (`gfx906:sramecc+:xnack-`)
* Build: Release, HIP 7.1.52802-9999
* Hardware state: approximately 930 MHz SCLK / 350 MHz MCLK; absolute rates
  are not canonical-clock claims
* Input: deterministic synthetic Q8_1 activation, seed `0x4D493050`
* Raw JSON artifacts: `bench/results/20260901T102000Z-389000/`

## Current production-like geometry

The selected current controls are `zero-point-128` for Gate/Up and
`zero-point-dot` (256-thread reduction) for Down.

| Projection | Shape | Median kernel | Mean kernel | Run mean spread | Effective BW | Oracle |
|---|---:|---:|---:|---:|---:|---|
| Gate | 12288×4096 | **50.240 µs** | 52.41 µs | 0.23 µs | 564.1 GB/s | 3/3 pass |
| Up | 12288×4096 | **50.240 µs** | 52.64 µs | 0.19 µs | 564.1 GB/s | 3/3 pass |
| Down | 4096×12288 | **57.280 µs** | 60.10 µs | 0.41 µs | 494.7–496.0 GB/s | 3/3 pass |

The three-projection kernel-only sum is approximately `157.76 µs` per layer,
or `5.68 ms` over 36 layers. This is only a shape-scaled primitive view and
must not replace the C7 production profile's approximately 7 ms FFN category.

## Available geometry controls

These existing controls were measured once as comparison points; they are not
promoted by this experiment.

| Projection | Current median | Wave64/alternate median | Result |
|---|---:|---:|---|
| Gate | 50.240 µs (`zero-point-128`) | **49.760 µs** (`zero-point-wave64`) | marginally faster |
| Up | 50.240 µs (`zero-point-128`) | **49.760 µs** (`zero-point-wave64`) | marginally faster |
| Down | **57.280 µs** (`zero-point-dot`) | 72.800 µs (`zero-point-wave64`) | materially slower |

The alternative timings remain within the same quantized oracle contract. The
roughly 1% Gate/Up Wave64 difference is not yet a sufficient end-to-end case
for a production change, and the existing Down alternative is a rejection for
this shape.

## Correctness

All recorded shape runs passed the Q4_0 × Q8_1 quantized oracle. No full-model
or generated-ID path changed. Existing Release correctness remains governed by
the M5-C7 path and its 19/19 CTest result.

## Decision

```text
BASELINE — characterize shapes before implementing C8b.
Do not promote an existing geometry control based on this marginal result.
```

## Follow-up

C8b should introduce one new gfx906 FFN GEMV geometry candidate, preserving the
Q4_0/Q8 activation contract and output behavior, then measure Gate/Up and Down
separately before any full-model promotion. The Down shape is the stronger
geometry discriminator because its current 256-thread path is materially
different from the available 128-thread/Wave64 alternatives.
