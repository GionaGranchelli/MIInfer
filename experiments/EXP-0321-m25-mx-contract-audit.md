# EXP-0321 — M25 pinned mx-llama contract audit

**Status:** RETEST; no new runtime code
**Milestone:** M25
**Date:** 2026-09-12
**Baseline:** `1bce8cb`
**Reference:** `/home/fedora-workstation/Development/mx-llama.cpp` at
`2e9d29fe736969160f17476ec6f0a6298cee6966`

## Hypothesis

The remaining P512 gap may come from a missing execution or data-layout
contract in the pinned gfx906 implementation, rather than from another
isolated arithmetic optimization.

## Motivation

The pinned reference is the primary external oracle for the current target.
This audit compares its Q4/Q5/Q6 repacked MMQ, Q8_1 quantization, GDN, and
Qwen3.8 recurrent-state contracts with the MIInfer implementations before
starting another optimization experiment.

The reference commit is MIT-licensed. Its repack source retains the upstream
copyright and license notices; MIInfer's adapted implementation is already
credited in `include/miinfer/qwen3_gpu_primitives.hpp`.

## Source comparison

| Reference contract | Pinned source | MIInfer equivalent | Result / risk |
| --- | --- | --- | --- |
| Q4_K/Q5_K/Q6_K compact repack | `ggml/src/ggml-cuda/q8_repack/repack-common.{cu,cuh}`, `repack-kernels.cuh`, `mul-mat*.cu` | `src/kquant_wave_layout.cpp`, `gfx906/kernels/kquant_wave_layout.hip` | The compact block contract and 64-row × 128-column × 4-subblock MMQ geometry match. MIInfer's staged legacy kernel remains the default. The reference register-prefetch pipeline was ported and measured 2.1–2.3× slower on MI50 in EXP-0304, so it is rejected. |
| Q8_1 MMQ activation blocks | `ggml/src/ggml-cuda/quantize.cu` (`quantize_mmq_q8_1`) | `launch_mx_q8_1_mmq_quantize()` and `MxQ8_1MmqBlock` | The 128-value block, 16-byte header, and 128-byte payload match. The M25-J batch workgroup variant was screened and rejected in EXP-0316; J remains explicitly disabled. |
| Affine Q4/Q5 dot product | `mmq.cuh` and repacked MMQ call sites | Mx staged MMQ in `kquant_wave_layout.hip` | MIInfer uses the same scale/min affine term and retained the interleaved DP4A order from EXP-0307. No untested replacement is warranted. |
| Chunked GDN | `ggml/src/ggml-cuda/gated_delta_net{,_chunk}.cu` | `gfx906/kernels/m12_gdn_chunk.hip`, `include/miinfer/m12_gdn_chunk.hpp` | The non-KDA recurrence and DPP/register scan are ported for the fixed Qwen3.8 shape. MIInfer transposes its canonical `[head][key][value]` state at the kernel boundary to the reference shard layout; this is deliberate and covered by the state oracle and EXP-0305. |
| Recurrent state ownership | Reference chunk kernel state `[S_v,S_v,H,n_seqs]` with transposed per-head output | MIInfer-owned `[head][key][value]` persistent state | The representations are not byte-compatible. A direct kernel copy would be wrong. The current boundary adaptation is the smallest contract-preserving approach; state cleanup remains a later phase. |

## Existing primitive evidence

The existing production-shape harness already covers the largest known matrix
differential. At B512, EXP-0303 measured current Mx compact MMQ against the
resident M23 path:

| Shape / type | M23 | Mx | M23/Mx |
| --- | ---: | ---: | ---: |
| gate/up, Q4_K, K=8192, N=6144 | 19,207.823 us | 7,852.154 us | 2.45× |
| ssm_out, Q5_K, K=8192, N=2048 | 7,120.314 us | 3,144.957 us | 2.26× |
| down, Q6_K, K=6144, N=8192 | 16,531.029 us | 8,149.594 us | 2.03× |

EXP-0307 retained the affine interleaved DP4A order after an additional
primitive comparison. EXP-0305 retained the GDN register scan after its
standalone and end-to-end A/B wins. These are ports of the relevant external
mechanisms, not merely similarly named kernels.

The reference trace in EXP-0300 still spends most GPU time in its Q4/Q5/Q6
repacked MMQ dispatches, but MIInfer's corresponding compact path is already
present and the reference register-pipeline variant is slower on the target
GPU. The current H/I path applies Mx attention packing to FFN and O while
leaving Q/K on the measured M23 path; M25-K's Q/K candidate was rejected in
EXP-0312 because its full-layer result was slower and used more VRAM.

## Interpretation

This audit found no missing literal source transplant with an evidence-backed
claim large enough to justify changing the P512 path now. The meaningful
differences are execution coverage, dispatch/materialization overhead, and
state ownership—not an unported Q4/Q5/Q6 block format or GDN recurrence.

The resident-M23 FFN repair in `1bce8cb` is required for H/I continuation
correctness, but it duplicates FFN representations and therefore must be
measured for VRAM and decode impact after a cold GPU reset.

## Decision

**KEEP** the current Mx MMQ, affine DP4A, and GDN ports.

**RETEST** H/I after privileged GPU reset. This document makes no new
performance claim and does not promote H/I to the default.

## Follow-up

1. Run the real continuation and same-process repeat check after reset, then
   rerun the full GPU CTest suite and the long-generation gate.
2. Record live/total/peak MIInfer allocation counters together with
   `hipMemGetInfo` at qualification boundaries.
3. Use the existing production-shape harness for the next candidate. Only
   proceed when a new candidate targets a measured differential not already
   covered by EXP-0303, EXP-0304, EXP-0305, EXP-0307, or EXP-0312.
4. Resolve the remaining source-vs-runtime question with the cold-state
   `3fbe0f1` versus `f38745e` P512 A/B before adding another optimization.
