# EXP-0006 — gfx906 Q4_0 × Q8_1 Packed-Dot Specialization

**Status:** REJECT  
**Milestone:** M2 — Prove Specialization  
**Author:** MIInfer project  
**Date:** 2026-08-30  
**Baseline commit:** `4d9d31ad67a8` (accepted EXP-0005 measurement)  
**Candidate commit:** `0543dd4ec714` (`gfx906: add q4 q8 packed dot candidate`)

## 1. Question

Does a gfx906-specific Q4_0 × Q8_1 GEMV using packed signed integer dot
products materially reduce latency on the real Qwen3-8B projection shapes?

## 2. Hypothesis

The scalar Q4 unpack/multiply path is instruction limited. Packing four signed
Q4-derived int8 values and four Q8 int8 values into
`v_dot4_i32_i8` operations should reduce instruction overhead while retaining
compressed Q4 weights in global memory.

## 3. Motivation

EXP-0005 was correctness-valid but reached only about 108–113 effective GB/s
on Q/O and FFN shapes despite the smaller Q4 representation. EXP-0006 isolates
the packed integer dot path; workgroup geometry, Q4/Q8 formats, matrix layout,
activation data, and the streaming-rotate-3 cache regime are unchanged. The
candidate does not use K-split, DPP, swizzle, inline ISA, expanded weights, or
custom persistent packing.

## 4. Source / Prior Art

The pinned reference is `milpster/gfx906-llama-cpp` at
`6e4ef6c1a553b8f61ad77bba18e9ca05aa677295`. Its quantized decode path is:

```text
ggml_cuda_mul_mat_vec_q → mmvq.cu → vec_dot_q4_0_q8_1
```

The reference uses `VDR_Q4_0_Q8_1_MMVQ = 2` and its own packed-dot helper.
EXP-0006 implements only the isolated MIInfer kernel mechanism; it does not
copy the reference runtime or layout.

## 5. Format and arithmetic contract

The accepted EXP-0005 format is retained:

| Format | Contract |
| --- | --- |
| Q4_0 | 32 values/block, 16 packed bytes, FP16 scale `d`; logical value `(nibble - 8) * d` |
| Q8_1 | 32 signed int8 values/block, FP16 scale `d`, FP16 sum metadata `s` |
| Accumulation | FP32 after each integer block dot |
| Output | FP16 |

The candidate decodes Q4 nibbles to signed `[-8, 7]` values in registers,
packs four values, consumes Q8 int8 values directly, and applies
`float(Q4.d) * float(Q8.d)` after the 32-value integer dot. Q8_1 `s` is not
mathematically required for this centered Q4_0 dot formulation, so it is not
read by the kernel. Q4 remains compressed in global memory.

## 6. Baseline and candidate

Both paths use one 256-thread workgroup per output row and the same 1 KiB LDS
FP32 reduction. The scalar baseline is the accepted EXP-0005 kernel. The
candidate changes only the inner dot operation to eight four-lane signed
integer dot operations per Q4_0 block, using ROCm Clang's
`__builtin_amdgcn_sdot4`.

Accepted scalar medians from EXP-0005:

| Shape | Q | K | V | O | G | U | D |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Median µs | 87.680 | 64.000 | 64.160 | 87.680 | 250.079 | 250.559 | 253.440 |

The clean candidate comparison was committed before measurement. Raw results
are retained locally under:

```text
bench/results/EXP-0006-q4-q8-packed-dot/accepted/
```

Each implementation has five repetitions for `gemv`, `one`,
`fanout-attention`, and `fanout-ffn`; each run retains environment-before,
result, telemetry, and environment-after files. The canonical GEMV and one
views use 10 warmups and 10,000 iterations; fan-out uses 10 warmups and 100
iterations. All accepted JSON records report commit `0543dd4ec714` and
`git_dirty=false`.

## 7. ISA proof

The installed ROCm Clang 20 compiler accepted
`__builtin_amdgcn_sdot4(int, int, int, bool)` for gfx906. The standalone
probe passed five signed-byte cases, including Q4-derived -8 and +7 values,
mixed signs, and the expected dot products. The extracted gfx906 code object
contains:

```text
v_dot4_i32_i8       # two static instructions in the GEMV candidate
v_dot4_i32_i8       # one additional instruction in the standalone probe
```

The candidate instructions were verified in the disassembly at offsets
`0x20A8` and `0x21B8`; the probe is at `0x2424`. The scalar Q4 GEMV code object
contains no `v_dot4_i32_i8`.

## 8. Correctness

Release and Debug CTest both passed:

```text
host-only
hip-smoke
fp16-gemv-correctness
q4-q8-gemv-correctness
q4-q8-dot4-probe
```

The Q4/Q8 test covers small indexing/tail shapes and all seven real shapes for
both scalar and packed paths. All 70 accepted GEMV records (five repetitions,
seven shapes, two implementations) report GPU-vs-quantized-oracle `PASS`, no
NaN/Inf, and cosine similarity `0.999999979`. Representative maximum absolute
errors are unchanged from the scalar path: Q/O `0.001926`, K/V `0.001125`,
G/U `0.001756`, and D `0.003163`. Quantized-vs-FP16 differences remain expected
quantization loss, not candidate failures.

## 9. Scalar versus packed-dot results

Values are the mean of five independent run medians in the streaming regime.
Delta is packed relative to scalar; negative is faster.

| Shape | Scalar µs | Packed-dot µs | Delta | Packed effective GB/s |
| --- | ---: | ---: | ---: | ---: |
| Q | 87.584 | 150.592 | +71.94% | 62.75 |
| K | 64.288 | 58.752 | -8.61% | 40.27 |
| V | 64.320 | 58.816 | -8.56% | 40.23 |
| O | 87.616 | 150.160 | +71.39% | 62.93 |
| G | 249.920 | 403.839 | +61.59% | 70.18 |
| U | 250.176 | 401.728 | +60.58% | 70.55 |
| D | 253.216 | 644.895 | +154.65% | 43.94 |

The K/V improvement is real but below the 25% major-regime threshold and is
not sufficient to offset the large Q/O and FFN regressions. The candidate
therefore fails the predefined packed-dot KEEP criterion.

## 10. FP16 control comparison

The best accepted FP16 controls are EXP-0002 for Q/O/G/U/D and EXP-0004
split-4 for K/V:

| Shape | Best FP16 µs | Packed Q4×Q8 µs | Packed relative to FP16 |
| --- | ---: | ---: | ---: |
| Q | 88.32 | 150.592 | +70.56% |
| K | 33.60 | 58.752 | +74.86% |
| V | 33.60 | 58.816 | +75.05% |
| O | 89.28 | 150.160 | +68.19% |
| G | 205.84 | 403.839 | +96.05% |
| U | 205.76 | 401.728 | +95.24% |
| D | 231.92 | 644.895 | +178.06% |

The current candidate has not converted Q4's lower logical weight footprint
into a latency advantage.

## 11. Quantization-inclusive views

The fixed Q8 quantizer remains the EXP-0005 implementation and costs a median
of 8.160 µs for length 4096 and 8.480 µs for length 12288. EXP-0006 did not
optimize it.

Five-run means of run medians:

| View | Scalar µs | Packed-dot µs | Delta |
| --- | ---: | ---: | ---: |
| one GEMV, Q | 91.168 | 165.808 | +81.91% |
| one GEMV, K | 67.456 | 61.984 | -8.11% |
| one GEMV, V | 67.424 | 62.112 | -7.88% |
| one GEMV, O | 91.200 | 164.464 | +80.11% |
| one GEMV, G | 256.304 | 419.183 | +63.56% |
| one GEMV, U | 255.456 | 417.823 | +63.56% |
| one GEMV, D | 258.496 | 648.159 | +150.93% |
| Q/K/V, quantize once | 221.744 | 294.560 | +32.85% |
| gate/up, quantize once | 488.415 | 827.695 | +69.46% |

The fan-out views confirm that quantizing once does not rescue the candidate.

## 12. Kernel resources and ISA observations

From the extracted gfx906 code-object metadata:

| Q4 GEMV kernel | VGPR | SGPR | LDS | spills | wavefront |
| --- | ---: | ---: | ---: | --- | ---: |
| Scalar | 13 | 21 | 1024 B | none | 64 |
| Packed-dot | 17 | 34 | 1024 B | none | 64 |

Both kernels have nine reduction barriers and the same 256-thread geometry.
The packed candidate's static function is 0x5fc bytes versus 0x320 bytes for
the scalar kernel. Its disassembly shows the intended dot instructions but
also substantial scalar bit extraction, shifts, ORs, and register packing
around them. This is consistent with an instruction/unpack cost or dot-path
throughput problem, not a missing ISA lowering. No scratch spills or global
INT8 weight expansion were observed.

Profiler-based hardware counters were not available in the validated ROCm
environment, so exact VALU/cache/HBM counters are `UNAVAILABLE`. Telemetry was
present in every accepted run: 1,839 records showed SCLK values including
1725 MHz and MCLK values including 1000 MHz, with junction temperature
32–92 C and reported socket power 19–237 W. The 237 W value is a transient
reported by the telemetry source above the 225 W cap and is retained rather
than hidden. No run was marked contaminated, no competing Ollama or
llama-server process was observed, and the large candidate regressions do not
depend on claiming a small timing win.

## 13. External MMVQ comparison

Direct isolated primitive timing for the pinned reference's MMVQ operation was
not available without adding a temporary reference-side harness. It is
therefore recorded as:

```text
EXTERNAL PRIMITIVE TIMING: UNAVAILABLE
```

The real external Q4 model-level reference remains valid: the pinned reference
successfully offloads the Q4_0 Qwen3-8B model and reports TG128 `91.411 t/s`
(approximately 10.94 ms/token). The source-supported external quantized path
is `ggml_cuda_mul_mat_vec_q` / `mmvq.cu`, not hipBLAS. EXP-0006 does not claim
direct primitive superiority over it.

## 14. Projection-only sanity check

Using the quantization-inclusive candidate medians:

```text
Q/K/V fan-out + O one-GEMV + gate/up fan-out + D one-GEMV
= 294.560 + 164.464 + 827.695 + 648.159 µs/layer
= 1,934.878 µs/layer
= 69.656 ms/token over 36 layers
```

This is a `PROJECTION-ONLY SANITY CHECK`, not a predicted runtime throughput;
it excludes attention, norms, RoPE, KV work, residuals, final projection, and
sampling. The accepted EXP-0005 scalar quantized projection-only value was
approximately 37.797 ms/token. Both are above the external Q4 model-level
TG128 total of approximately 10.94 ms/token, so this candidate is not an
end-to-end threat to the reference.

## 15. Bottleneck interpretation

| Regime | Evidence | Classification | Confidence |
| --- | --- | --- | --- |
| K/V | Packed path improves 8.6–8.6% but remains slower than EXP-0004 split-4 FP16; same geometry and low row count | mixed: grid/parallelism limited with packed instruction overhead | MEDIUM |
| Q/O | Packed path is 71% slower; 17 vs 13 VGPR and 34 vs 21 SGPR; expanded candidate instruction body | instruction/unpack limited, with resource overhead | MEDIUM |
| G/U | Packed path is 61–62% slower despite large row grid and no spills | instruction/unpack or packed-dot throughput limited | MEDIUM |
| D | Packed path is 155% slower with the larger K reduction | mixed: instruction/unpack and reduction/loop cost | MEDIUM |

## 16. Decision

```text
EXP-0006 REJECT
```

This rejects the tested straightforward register-pack/unpack implementation as
a performance candidate. It does not prove that every gfx906 packed-dot design
is unhelpful. The ISA mechanism is available and correct, but this composition
does not satisfy the measured latency criterion.

M2 remains open because direct MMVQ primitive timing is unavailable and the
candidate is not competitive with the accepted FP16 controls or the external
model-level Q4 reference.

## 17. Next experiment

Recommend exactly one follow-up:

```text
EXP-0007 — reduce Q4 register-unpack/pack overhead in a packed-dot kernel
```

The next experiment should isolate a lower-overhead Q4 representation or
kernel-native consumption strategy, retain compressed global weights, and
compare against this frozen candidate and EXP-0005. Do not implement it as
part of EXP-0006.
