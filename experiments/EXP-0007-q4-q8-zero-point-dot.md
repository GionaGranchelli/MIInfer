# EXP-0007 — gfx906 Zero-Point-Corrected Q4_0 × Q8_1 Dot4

**Status:** KEEP  
**Milestone:** M2 — Prove Specialization  
**Author:** MIInfer project  
**Date:** 2026-08-30  
**Baseline commit:** `4d9d31ad67a8` (accepted EXP-0005 measurement)  
**Candidate commit:** `4842c9dd77ec` (`gfx906: add zero point corrected q4 q8 dot`)

## 1. Question

Can Q4_0 nibbles remain unsigned in gfx906 packed dot4 operands, with Q8_1
sum metadata supplying the Q4 zero-point correction, and thereby remove the
overhead that caused EXP-0006 to regress?

## 2. Hypothesis and motivation

EXP-0006 proved `v_dot4_i32_i8` is available and correct, but its signed Q4
conversion required repeated extraction, subtract, and register packing. The
pinned reference instead dots Q4 nibbles in the `[0,15]` range and applies the
zero point algebraically. This experiment changes only that inner Q4/Q8 dot
and correction path. Geometry, reduction, storage format, activation data,
and streaming-rotate-3 cache regime remain fixed.

## 3. Algebra and format contract

For Q4 nibble `n`, Q8 value `q`, and stored scales:

```text
w = d4 × (n - 8)
a = d8 × q
Σ(w × a) = d4 × [d8 × Σ(n × q) - 8 × d8 × Σq]
```

Q8_1 stores `s8 = d8 × Σq` as FP16 metadata, so the candidate computes:

```text
d4 × [d8 × raw_integer_dot - 8 × s8]
```

The candidate consumes `Q8_1.d`, `Q8_1.s`, and `Q8_1.qs`. Q4 remains the
canonical 18-byte Q4_0 block in global memory; no expanded INT8 weights or new
`.mi50` layout is used.

The CPU identity tests cover Q4 0/15, logical -8/+7, positive/negative and
zero-sum Q8, mixed signs, Q8 ±127, and deterministic random blocks. The ideal
algebraic identity has maximum error `1.53e-5`; use of stored FP16 `s8` has a
maximum observed test deviation of `0.02214` from the unrounded identity.

## 4. Implementations

Three paths were retained and benchmarked:

```text
scalar          EXP-0005 scalar nibble unpack and FP32 products
packed-dot      EXP-0006 signed Q4 register-pack dot4 control (REJECT)
zero-point-dot  EXP-0007 unsigned Q4 dot4 plus Q8_1.s correction
```

All use one 256-thread workgroup per row, 1 KiB LDS, and the same reduction.
No K-split, DPP, swizzle, prefetch, fusion, or custom weight layout was added.

Raw artifacts are retained locally under:

```text
bench/results/EXP-0007-q4-q8-zero-point-dot/accepted/
```

There are five independent repetitions for GEMV, one-GEMV, both fan-outs, and
the size-matched memory reference. Accepted records were captured from clean
commit `4842c9dd77ec` with `git_dirty=false`, 10 warmups, and 10,000 measured
iterations for GEMV/one views. Fan-out used 100 iterations.

## 5. ISA and resource comparison

The extracted gfx906 code object for the candidate contains eight
`v_dot4_i32_i8` instructions, eight `v_and_b32` mask operations, four shifts,
and no OR-based signed-byte repacking. It has nine reduction barriers.

| Property | Scalar | EXP-0006 signed-pack | EXP-0007 zero-point |
| --- | ---: | ---: | ---: |
| `v_dot4_i32_i8` | 0 | 2 static | 8 |
| Q4 mask ops | — | substantial | 8 |
| shifts | — | substantial | 4 |
| OR/pack ops | — | substantial | 0 observed |
| VGPR | 13 | 17 | 23 |
| SGPR | 21 | 34 | 19 |
| LDS | 1024 B | 1024 B | 1024 B |
| spills | none | none | none |
| barriers | 9 | 9 | 9 |
| static body bytes | 0x320 | 0x5fc | 0x3bc |

The candidate has more VGPRs than either control, but no scratch spills. Its
static body is substantially smaller than EXP-0006 because signed lane
materialization is eliminated.

## 6. Build and correctness

Debug and Release builds and CTest passed all five tests:

```text
host-only
hip-smoke
fp16-gemv-correctness
q4-q8-gemv-correctness
q4-q8-dot4-probe
```

All seven real shapes pass GPU-vs-quantized-oracle correctness for all three
implementations. For zero-point-dot, maximum absolute error by regime is:

| Shapes | Max abs | Mean abs | Cosine |
| --- | ---: | ---: | ---: |
| Q/O | 0.002610 | 0.000477 | 0.999999897 |
| K/V | 0.002668 | 0.000596 | 0.999999843 |
| G/U | 0.002754 | 0.000549 | 0.999999865 |
| D | 0.004122 | 0.000894 | 0.999999879 |

No NaN or Inf was observed. Differences from the scalar path are consistent
with using the stored FP16 Q8_1 `s` metadata and remain within the accepted GPU
tolerance.

## 7. GEMV performance

Five-run means of per-run medians in the streaming regime:

| Shape | Scalar µs | Signed-pack µs | Zero-point µs | Δ vs scalar | Zero-point GB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q | 91.200 | 150.592 | 29.088 | -68.11% | 325.0 |
| K | 67.808 | 58.768 | 56.224 | -17.08% | 42.1 |
| V | 67.776 | 58.720 | 56.800 | -16.19% | 41.6 |
| O | 91.200 | 150.112 | 29.056 | -68.14% | 325.6 |
| G | 253.567 | 419.807 | 63.840 | -74.82% | 443.8 |
| U | 253.424 | 418.000 | 63.840 | -74.81% | 443.8 |
| D | 258.399 | 648.799 | 61.440 | -76.22% | 462.7 |

The candidate exceeds the 25% latency-reduction criterion on Q/O, G/U, and D.
K/V improve, but remain the known small-grid outlier and remain slower than
EXP-0004 split-4 FP16 controls.

Against the best accepted FP16 controls, zero-point-dot is approximately 67%
faster on Q/O, 69% faster on G/U, and 74% faster on D. It is approximately
67–69% slower than split-4 FP16 on K/V.

## 8. Quantization-inclusive views

Five-run means of per-run medians:

| View | Scalar µs | Signed-pack µs | Zero-point µs |
| --- | ---: | ---: | ---: |
| One GEMV, Q | 91.200 | 166.192 | 29.088 |
| One GEMV, K | 67.808 | 62.080 | 56.224 |
| One GEMV, V | 67.776 | 62.176 | 56.800 |
| One GEMV, O | 91.200 | 164.656 | 29.056 |
| One GEMV, G | 253.567 | 419.807 | 63.840 |
| One GEMV, U | 253.424 | 418.000 | 63.840 |
| One GEMV, D | 258.399 | 648.799 | 61.440 |
| Q/K/V, quantize once | 216.304 | 296.816 | 128.463 |
| Gate/up, quantize once | 501.263 | 829.535 | 125.408 |

The fixed Q8 quantizer remains approximately 8.16 µs at length 4096 and 8.48
µs at length 12288. The faster GEMV makes quantization material in fan-out,
but it does not erase the candidate's large total gain.

## 9. Size-matched memory reference

The diagnostic contiguous copy benchmark measured the following empirical
read+write throughput, not peak HBM bandwidth:

| Bytes | Median µs | Effective GB/s |
| ---: | ---: | ---: |
| 2.25 MiB | 20.416 | 231.1 |
| 9 MiB | 50.752 | 371.9 |
| 27 MiB | 135.488 | 417.9 |

Zero-point-dot reaches approximately 325.0–325.6 GB/s on 9 MiB Q/O weights and
443.8 GB/s on 27 MiB G/U weights. These are logical effective bandwidths, not
profiler-confirmed HBM traffic. D reaches 462.7 GB/s because its K dimension
and launch work differ from G/U.

## 10. Hardware validity

Telemetry was retained for every accepted run. Across 2,321 records, observed
SCLK values included 1725 MHz and MCLK values included 1000 MHz; junction
temperature ranged 33–90 C and reported socket power ranged 19–240 W. The
transient power reading above the 225 W cap is retained. No run was marked
contaminated and no competing Ollama or llama-server workload was observed.

The accepted result is robust across the five repetitions; the candidate's
major-regime improvements are much larger than the observed run-to-run timing
variation.

## 11. External MMVQ comparison

Direct isolated primitive timing for the pinned reference remains:

```text
EXTERNAL MMVQ PRIMITIVE TIMING: UNAVAILABLE
```

The external quantized path is confirmed from source as
`ggml_cuda_mul_mat_vec_q` → `mmvq.cu` → `vec_dot_q4_0_q8_1`. The existing
Qwen3-8B Q4_0 model-level reference remains `TG128 = 91.411 t/s`, or about
10.94 ms/token total. EXP-0007 therefore does not claim direct primitive
superiority or M2 GO.

## 12. Projection-only sanity check

Using the quantization-inclusive candidate medians:

```text
Q/K/V fan-out + O one-GEMV + gate/up fan-out + D one-GEMV
= 128.463 + 29.056 + 125.408 + 61.440 µs/layer
= 344.367 µs/layer
= 12.397 ms/token over 36 layers
```

This is a `PROJECTION-ONLY SANITY CHECK`, not predicted model throughput. It
excludes attention, norms, RoPE, KV operations, residuals, final projection,
and sampling. It improves substantially over EXP-0005's approximately 37.797
ms/token and EXP-0006's approximately 69.656 ms/token, but remains above the
external Q4 model-level total of approximately 10.94 ms/token.

## 13. Decision

```text
EXP-0007 KEEP
```

The standard Q4_0 layout can feed correct gfx906 dot4 operations efficiently
when the Q4 zero point is corrected through Q8_1 metadata. This is a material
specialization win over the accepted scalar baseline and the EXP-0006 signed
packing control.

## 14. M2 status

```text
M2 OPEN
```

This is a strong M2 signal, but direct MMVQ primitive timing is still
unavailable and the candidate has not been integrated into a complete runtime.
Those limitations prevent an M2 GO declaration.

## 15. Next experiment

Recommend exactly one:

```text
EXP-0008 — compose zero-point dot4 with split-K for K/V
```

K/V remains the only major outlier: zero-point-dot improves the scalar path but
is still slower than the accepted EXP-0004 split-4 FP16 control. The next test
should compose the accepted zero-point dot4 inner operation with K-split while
keeping the composition as a separate experiment. Do not implement that
composition in EXP-0007.
