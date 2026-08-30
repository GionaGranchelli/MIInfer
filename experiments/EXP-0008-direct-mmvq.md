# EXP-0008 — Direct MIInfer vs gfx906 llama.cpp MMVQ

**Status:** KEEP  
**Milestone:** M2 — Prove Specialization  
**Author:** MIInfer project  
**Date:** 2026-08-30  
**Baseline commit:** `4842c9dd77ec` (accepted EXP-0007 kernel)  
**Reference commit:** `6e4ef6c1a553b8f61ad77bba18e9ca05aa677295`

## 1. Question

How does the accepted MIInfer EXP-0007 zero-point Q4_0 × Q8_1 GEMV compare
directly with the pinned gfx906 llama.cpp MMVQ primitive on the exact seven
Qwen3-8B projection shapes?

## 2. Hypothesis and motivation

EXP-0007 materially beat MIInfer's scalar quantized baseline, but that was not
yet evidence against the strongest relevant gfx906 implementation. This
experiment measures the production reference MMVQ path directly before any
further MIInfer optimization or runtime work.

## 3. Reference path

The pinned checkout was clean at:

```text
/home/fedora-workstation/Development/mi50-artifacts/milpster-gfx906-reference-6e4ef6c
6e4ef6c1a553b8f61ad77bba18e9ca05aa677295
```

The launcher invokes the exported low-level entry
`ggml_cuda_op_mul_mat_vec_q`, which dispatches through
`mul_mat_vec_q_switch_type` to the Q4_0 MMVQ kernel. This excludes the
high-level FP32-to-Q8 quantization step and does not reimplement the dot
operation. The reference source confirms the Q4_0 path is
`vec_dot_q4_0_q8_1_impl` in `ggml/src/ggml-cuda/vecdotq.cuh`.

The existing `test-backend-ops` facility was inspected first. It can time
generic `MUL_MAT` cases, but does not expose these exact seven batch-1 shapes
as a primitive-only operation, so a temporary launcher was used in the
separate reference checkout. The launcher was removed after compilation and
was not added to MIInfer or the reference repository.

## 4. Semantic and timing contract

Both sides use the pinned standard Q4_0 and Q8_1 block representations and
FP32 accumulation. Both consume an already-quantized Q8_1 activation, so Q8
activation quantization is outside the primitive timing. Allocation, copies,
quantization, handle creation, and tensor setup are outside timing.

The MIInfer output is FP16; the low-level reference destination is FP32. This
is recorded as a semantic difference. The comparison is therefore a raw
primitive latency comparison, not a claim of bit-identical output contracts.
The reference launcher did not copy the output back for an independent
elementwise check; reference correctness is inherited from the pinned
production path and its existing Q4 model smoke test.

The reference launcher allocated three equivalent Q4 weight buffers and
rotated them on successive launches. Its result records label this
`streaming-rotate-3`, matching the accepted MIInfer cache methodology rather
than repeatedly using one small K/V matrix from the same address.

## 5. Shapes

```text
Q  4096 × 4096
K  1024 × 4096
V  1024 × 4096
O  4096 × 4096
G 12288 × 4096
U 12288 × 4096
D  4096 × 12288
```

Each side used 10 warmups and 20,000 HIP-event samples per repetition, with
five repetitions per shape. Raw artifacts are retained locally under:

```text
bench/results/EXP-0008-direct-mmvq/
```

The reference artifacts are under `reference/<shape>/repetition-N/` and the
same-protocol current MIInfer rerun is under
`miinfer-exp7-current/<shape>/repetition-N/`.

## 6. MIInfer control

The MIInfer side is the unchanged EXP-0007 zero-point-dot kernel:

```text
one 256-thread workgroup per row
Q4_0 remains compressed in global memory
Q8_1.d, Q8_1.s, and Q8_1.qs consumed directly
v_dot4_i32_i8 with algebraic Q4 zero-point correction
```

No K-split, new layout, DPP, swizzle, prefetch, or other candidate was added.
The executable embeds `4842c9dd77ec` and reports `git_dirty=false`.

## 7. Results

Values below are the means of the five per-run medians from the matched
20,000-sample reruns. Delta is `(MIInfer - reference) / reference`; negative
means MIInfer is faster.

| Shape | MIInfer EXP-0007 µs | llama.cpp MMVQ µs | MIInfer delta | Reference effective GB/s |
| --- | ---: | ---: | ---: | ---: |
| Q | 25.600 | 26.176 | -2.20% | 362.7 |
| K | 52.768 | 13.120 | +302.20% | 180.5 |
| V | 52.704 | 13.120 | +301.71% | 180.5 |
| O | 25.728 | 26.272 | -2.07% | 360.4 |
| G | 60.320 | 61.600 | -2.08% | 460.5 |
| U | 60.352 | 61.632 | -2.08% | 460.5 |
| D | 57.600 | 64.928 | -11.29% | 436.3 |

The reference effective bandwidth uses Q4 weight bytes + Q8 activation bytes +
FP32 output bytes. It is a logical metric, not measured HBM traffic.

Regime classification using a ±5% equivalence band:

| Regime | Result |
| --- | --- |
| Q/O | approximately equivalent; MIInfer is 2.1–2.2% faster |
| K/V | reference materially faster, about 4.0× |
| G/U | approximately equivalent; MIInfer is about 2.1% faster |
| D | MIInfer materially faster, 11.3% |

The major result is mixed: the direct comparison does not show a material
MIInfer advantage across both Q/O and FFN families. It does show a clear D
advantage and identifies K/V as the remaining large gap.

The per-run median ranges were MIInfer: Q 25.600–25.600, K 52.640–52.960,
V 52.640–52.800, O 25.600–25.760, G 60.320–60.320, U 60.320–60.480, and D
57.600–57.600 µs. Reference ranges were Q 26.080–26.400, K/V 13.120–13.120,
O 26.240–26.400, G 61.600–61.600, U 61.600–61.760, and D 64.800–64.960 µs.
The small ranges support stable timing; they do not remove the semantic
FP16-output versus FP32-output difference documented above.

## 8. Historical control discrepancy

The accepted EXP-0007 document reports medians of Q 29.088, K 56.224, V
56.800, O 29.056, G/U 63.840, and D 61.440 µs. The retained raw EXP-0007
artifacts at commit `4842c9dd77ec` instead aggregate to approximately Q/O
25.600, K/V 52.8, G/U 60.5, and D 57.8 µs. This experiment does not rewrite
that historical record or select the more favorable set. It records a fresh
same-protocol MIInfer control alongside the fresh reference measurements.

The fresh control is used for the direct table because it was measured in the
same session/protocol and has five clean repetitions. The discrepancy should
be reconciled before future end-to-end claims, but it does not invalidate the
EXP-0007 semantic or correctness result.

## 9. Hardware validity

All runs used the M0-approved gfx802-isolated state. Environment captures
reported clean MIInfer commit state and the reference stderr identified one
gfx906 device:

```text
AMD Instinct MI60 / MI50, gfx906:sramecc+:xnack-
```

Sustained active reference samples reached the expected 1725 MHz SCLK and
1000 MHz MCLK. Transient startup samples sometimes showed lower clocks before
the active state; the measured operation medians were stable. Across accepted
reference telemetry, junction temperature reached approximately 76 C and
reported socket power reached approximately 230 W. No competing Ollama,
llama-server, or other workload was observed in the run records. The pinned
reference environment was clean at the required commit.

## 10. Reference ISA and resources

For gfx906, the pinned GCN parameter table selects two waves per output row
for `ncols_dst=1`; the launch is 64 threads × 2 and one row per block. The
Q4_0 path uses `VDR_Q4_0_Q8_1_MMVQ=2`. The source's final reduction combines
the two warp partials through shared memory and a warp reduction.

The extracted Q4_0, two-wave, no-fusion code object is:

```text
libggml-hip.so.33.hipv4-amdgcn-amd-amdhsa--gfx906
```

The exact kernel symbol has:

```text
static body: 0x678 bytes
VGPR: 40
numbered SGPR: 28
private/scratch: 0
LDS: 256 B from one nwarps-1 partial row
barriers: 1
static v_dot4_i32_i8: 16
static v_lshr: 6
static v_and: 11
```

The corresponding MIInfer EXP-0007 observation is:

```text
static body: 0x3bc bytes
VGPR: 23
SGPR: 19
LDS: 1024 B
spills: none
barriers: 9
static v_dot4_i32_i8: 8
```

Static instruction counts are not dynamic counts. The resource difference
shows that the reference and MIInfer are not the same kernel geometry even
though both use the same Q4/Q8 algebra and gfx906 dot instruction.

## 11. Architectural differences relevant to K/V

The reference explicitly selects two waves per row for batch-1 MMVQ on GCN,
while the MIInfer control uses four waves per row. The reference's K/V result
is therefore evidence that its work distribution and reduction geometry are
better suited to `M=1024`; it is not evidence that Q4 dot4 is intrinsically
faster in the reference. A split-K composition remains a bounded follow-up
hypothesis, but it is not implemented here.

For Q/O/G/U/D, the two kernels are close under the matched protocol. The
reference's Q4 path uses unsigned nibble mask/shift operations and algebraic
zero-point correction through Q8_1 sum metadata, the same central mechanism
validated by EXP-0007. The remaining differences are primarily geometry,
load organization, reduction, and resource trade-offs; this experiment does
not attribute causality beyond the evidence above.

## 12. M2 decision

```text
EXP-0008 KEEP
```

A trustworthy direct primitive comparison is established and its raw
artifacts are retained. However:

```text
M2 OPEN
```

The direct evidence is mixed rather than a material MIInfer win across both
Q/O and G/U/D. MIInfer wins D by 11.3% and is approximately tied on Q/O/G/U,
but the pinned MMVQ path wins K/V by about 4×. This is not sufficient for
`M2 GO` under the project's strongest-competitor gate.

## 13. Next experiment

Recommend exactly one:

```text
EXP-0009 — compose zero-point dot4 with split-K for K/V
```

The target is the observed K/V gap. Keep the EXP-0007 inner dot unchanged and
measure the complete split-K operation, including partial-result reduction,
against both the current MIInfer control and the reference MMVQ result. Do
not begin that composition in EXP-0008.

## 13.1 Re-evaluation after EXP-0009

The `M2 OPEN` decision above is historical for the geometry available when
EXP-0008 was run. EXP-0009 subsequently tested the measured MMVQ difference:
128 threads per row removed the idle-thread/reduction mismatch and brought
K/V below MMVQ latency while preserving or improving the other major shapes.
The later decision is recorded in EXP-0009 as:

```text
EXP-0009 KEEP
M2 GO
```
