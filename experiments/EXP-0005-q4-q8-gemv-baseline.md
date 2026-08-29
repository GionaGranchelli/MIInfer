# EXP-0005 — Q4_0 × Q8_1 Quantized GEMV Baseline

**Status:** KEEP  
**Milestone:** M1  
**Author:** MIInfer project  
**Date:** 2026-08-30  
**Baseline commit:** `4d9d31ad67a8`  
**Candidate commit:** `4d9d31ad67a8`

## 1. Question

What correctness, quantization-quality, and performance baseline does a
straightforward Q4_0-weight × Q8_1-activation GEMV establish for the seven
real Qwen3-8B projection shapes on MI50/gfx906?

This experiment freezes the quantized contract. It does not implement
gfx906-specific packed-dot instructions or other optimization candidates.

## 2. Hypothesis

A simple scalar-unpack Q4_0 × Q8_1 HIP kernel will be correctness-valid, but
will leave substantial performance on the table compared with the pinned
reference's optimized quantized path. Measuring it provides a stable target
for the first gfx906 specialization experiment.

## 3. Baseline and scope

The tested contract is:

```text
weights:      Q4_0
activations:  Q8_1
accumulator:  FP32
output:       FP16
```

The project-owned kernel uses one 256-thread workgroup per output row, scalar
Q4 unpacking, FP32 accumulation, and an LDS tree reduction. Device allocation,
copies, quantized-weight creation, and handle/stream setup are outside the
timed region. The canonical matrix regime is STREAMING with three rotated
device weight buffers.

The implementation deliberately does not use `v_dot4_i32_i8`, inline ISA,
DPP, `ds_swizzle`, half-wave specialization, custom packing, prefetch,
autotuning, fusion, graphs, or runtime integration.

## 4. Format semantics

The format is compatible with the pinned reference
`milpster/gfx906-llama-cpp` at commit
`6e4ef6c1a553b8f61ad77bba18e9ca05aa677295`, using its
`ggml/src/ggml-common.h` and `ggml/src/ggml-quants.c` definitions.

### Q4_0

```text
block size:        32 values
storage:           half d + 16 bytes qs = 18 bytes/block
packing:           qs[0..15] low nibbles for values 0..15;
                   qs[0..15] high nibbles for values 16..31
scale:             d = max(values) / -8, FP16
dequantization:    (q - 8) * d
```

The host implementation uses the reference rounding rule
`q = clamp((int8_t)(x / d + 8.5), 0, 15)` after selecting the signed block
maximum. It uses the exact stored FP16 source values.

### Q8_1

```text
block size:        32 values
storage:           half d + half s + 32 int8 qs = 36 bytes/block
scale:             d = amax / 127, FP16
values:            q = round(x / d), clamped to [-127, 127]
metadata:          s = FP16(sum(q) * d)
```

The GPU quantizer matches all Q8 integer values and scales. The `s` metadata
is accepted within 0.0011 absolute error, which covers the observed one-FP16-
ULP CPU/GPU reduction-rounding difference. The largest observed difference was
0.000976562.

## 5. Correctness method

The CPU oracle reconstructs the value represented by the actual Q4_0 and Q8_1
blocks and accumulates in FP32. GPU results are compared with that oracle.
Separately, the quantized oracle is compared with the FP16 oracle to report
quantization quality loss. The latter is intentionally not a GPU correctness
pass/fail metric; Q4 quantization is expected to differ from FP16.

The test covers small indexing/tail shapes `{7,32}` and `{257,64}`, followed
by all seven real shapes. Debug and Release CTest both pass.

## 6. Benchmark protocol

```text
platform:          M0-approved gfx802-isolated MI50/gfx906
commit:            4d9d31ad67a8
dirty:             false
warmups:           10
iterations:        10,000 for GEMV and one-GEMV modes
                   10,000 for activation quantization
                   100 for fan-out modes
repetitions:       5
cache regime:      STREAMING, three rotated weight buffers
telemetry:         100 ms sampling, retained per run
```

Raw run directories are retained locally under:

```text
bench/results/EXP-0005-q4-q8/accepted/
```

Each accepted invocation retains `environment-before.json`, `result.json`,
`telemetry.jsonl`, and `environment-after.json`. Earlier failed/superseded
pilot artifacts remain under the experiment result root and are not included
in the accepted aggregates.

## 7. Activation quantization

The standalone GPU FP16 → Q8_1 quantization results are the median of five
run medians:

| Length | Median µs | Effective input+output GB/s | Correctness |
| ---: | ---: | ---: | --- |
| 4096 | 8.160 | 1.569 | PASS |
| 12288 | 8.480 | 4.528 | PASS |

The effective I/O value includes the FP16 input and Q8_1 output bytes. It is
not a claim about HBM traffic.

## 8. Quantized GEMV results

The following are medians of the five independent per-run medians. Logical
bytes are Q4_0 weight bytes + Q8_1 activation bytes + FP16 output bytes.

| Shape | M | K | Median µs | Effective GB/s | GPU vs quantized oracle | GPU max abs | Quantized-vs-FP16 cosine |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| Q | 4096 | 4096 | 87.680 | 107.86 | PASS | 0.001926 | 0.997820 |
| K | 1024 | 4096 | 64.000 | 36.97 | PASS | 0.001125 | 0.997986 |
| V | 1024 | 4096 | 64.160 | 36.86 | PASS | 0.001125 | 0.997986 |
| O | 4096 | 4096 | 87.680 | 107.86 | PASS | 0.001926 | 0.997820 |
| G | 12288 | 4096 | 250.079 | 113.47 | PASS | 0.001756 | 0.997866 |
| U | 12288 | 4096 | 250.559 | 113.32 | PASS | 0.001756 | 0.997866 |
| D | 4096 | 12288 | 253.440 | 111.88 | PASS | 0.003163 | 0.997867 |

GPU-vs-oracle cosine similarity was 0.999999979 for every real shape. The
quantized-vs-FP16 maximum absolute errors were 0.275463 (K/V), 0.363907 (Q/O),
0.364469 (G/U), and 0.575042 (D); these are quantization quality observations,
not implementation failures.

Q4_0 storage is 18 bytes per 32 weights, or 3.5556× smaller than FP16 before
any model-level metadata. The real-shape weight sizes are 2.25 MiB for K/V,
9.00 MiB for Q/O, and 27.00 MiB for G/U/D.

## 9. Quantization-inclusive views

Mode `one` measures FP16 activation → Q8_1 followed by one GEMV. Its aggregate
medians are:

| Shape | Median µs |
| --- | ---: |
| Q | 91.199 |
| K | 67.840 |
| V | 67.840 |
| O | 91.200 |
| G | 256.000 |
| U | 256.480 |
| D | 257.919 |

The measured fan-out modes quantize once and reuse Q8_1 activations:

| Fan-out | Median µs | Correctness |
| --- | ---: | --- |
| Q/K/V, one Q8 quantization | 212.160 | PASS |
| Gate/Up, one Q8 quantization | 488.640 | PASS |

These are whole logical operations, not sums of separately timed kernels.
They are the relevant measurements for later static activation reuse.

## 10. Comparison with accepted FP16 controls

| Shape | FP16 plain µs | FP16 split-4 control µs | Q4_0 × Q8_1 baseline µs |
| --- | ---: | ---: | ---: |
| Q | 88.32 | — | 87.680 |
| K | 70.72 | 33.60 | 64.000 |
| V | 70.72 | 33.60 | 64.160 |
| O | 89.28 | — | 87.680 |
| G | 205.84 | — | 250.079 |
| U | 205.76 | — | 250.559 |
| D | 231.92 | — | 253.440 |

The split-4 FP16 values are only accepted for K/V; other shapes retain their
plain FP16 controls. The straightforward quantized kernel is not yet a
latency win over those FP16 controls. This is expected: the baseline performs
scalar unpacking and does not yet exploit gfx906 packed integer dot products.

## 11. Projection-only sanity check

Using the measured whole-operation fan-outs for Q/K/V and Gate/Up, plus the
quantization-inclusive O and D values:

```text
Q/K/V fan-out       212.160 µs
O                    91.200 µs
Gate/Up fan-out      488.640 µs
D                   257.919 µs
--------------------------------
per layer          1,049.919 µs
× 36 layers       37.797 ms/token
projection-only    ~26.46 tok/s ceiling
```

This is a **PROJECTION-ONLY SANITY CHECK**, not a predicted runtime result. It
excludes attention, normalization, RoPE, KV work, residuals, activations, the
final projection, sampling, launch/fusion effects, and any runtime overhead.
For comparison, the accepted FP16 projection-only estimate was about 31.98
ms/token (~31.3 tok/s), the pinned F16 reference measured 31.60 tok/s TG128,
and the pinned Q4_0 reference measured 91.41 tok/s TG128.

## 12. External Q4_0 reference

The pinned reference checkout remained at commit
`6e4ef6c1a553b8f61ad77bba18e9ca05aa677295`. The F16 source artifact was
unchanged:

```text
source revision: Qwen/Qwen3-8B b968826d9c46dd6066d109eabc6255188de91218
F16 SHA256:      c1fd1fc17831ebc0001d81c97a3f78626dd1f977841dec532eef60177abb2a1c
```

Conversion used the pinned checkout's `llama-quantize` built from the same
commit:

```bash
llama-quantize Qwen3-8B-f16-b968826d.gguf \
  Qwen3-8B-q4_0-b968826d.gguf Q4_0 2
```

The quantizer reported 4.66 BPW including block overhead. The artifact is
4,768,792,576 bytes (4.5 GiB) with SHA256:

```text
458634762bea7dbe19f3ce0614465bafd15ee90e815229c427043afcf195d628
```

The reference Q4_0 smoke test generated valid text and reported `offloaded
37/37 layers to GPU`, with a 4,214.03 MiB ROCm0 model buffer. Five-repetition
canonical measurements with batch 2048 and ubatch 512 were:

| Workload | Mean t/s | Stddev | Raw samples t/s |
| --- | ---: | ---: | --- |
| PP512 | 987.100 | 13.576 | 963.315, 994.689, 995.423, 993.643, 988.427 |
| PP2048 | 951.493 | 0.810 | 950.469, 951.618, 950.867, 952.205, 952.306 |
| TG128 | 91.411 | 0.351 | 90.813, 91.688, 91.395, 91.597, 91.562 |

The source-supported quantized decode path is the pinned reference's
`ggml_cuda_mul_mat_vec_q` / `mmvq.cu` path. For Q4_0 it dispatches
`vec_dot_q4_0_q8_1`, uses `VDR_Q4_0_Q8_1_MMVQ = 2`, and selects the GCN
parameter table for gfx906. This is the relevant external quantized competitor;
hipBLAS is not the reference Q4 decode path.

## 13. Decision

**EXP-0005 KEEP**

The format semantics are frozen, all seven real shapes pass GPU-vs-quantized-
oracle correctness, quantization quality is separately characterized,
activation quantization and fan-out costs are measured, five-run raw artifacts
are retained on a clean benchmark commit, and the pinned Q4_0 model/reference
executes with complete GPU offload and PP/TG baselines.

KEEP means this is the stable quantized target for later specialization. It
does not mean the project-owned scalar kernel is optimized.

## 14. Follow-up

Recommend exactly one next experiment:

> **EXP-0006 — gfx906 Q4_0 × Q8_1 packed-dot specialization**, beginning with
> register unpack/conversion and `v_dot4_i32_i8`, with correctness and whole
> GEMV timing retained as the acceptance gates.

Do not implement that candidate as part of EXP-0005.
