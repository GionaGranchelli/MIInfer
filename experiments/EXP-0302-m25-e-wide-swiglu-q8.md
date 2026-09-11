# EXP-0302 — M25-E wide SwiGLU to MMQ-Q8 producer

**Status:** REJECT
**Milestone:** M25-E
**Date:** 2026-09-11

## Hypothesis

Computing wide-prefill SwiGLU directly into the existing `M23Q8_1MmqBlock`
stream will remove the intermediate FP32 activation read/write and the
standalone MMQ-Q8 quantizer launch.

## Baseline and candidate

- Baseline: `launch_qwen3_silu_mul` followed by
  `launch_m23_q8_1_mmq_quantize`.
- Candidate: one 128-thread kernel computes the same SiLU multiply and emits
  the existing four-group-per-128 MMQ-Q8 block layout.
- The candidate was opt-in through
  `MIINFER_PREFILL_WIDE_FUSED_SWIGLU_Q8=1`; decode was untouched.

This is an MIInfer-native mapping experiment. It did not copy external code.

## Environment

- AMD Instinct MI50 / gfx906, spot-checked at SCLK/MCLK 1606/1000 MHz
- Qwen3.8-27B-Q4_K_M GGUF, model SHA from EXP-0285
- exact 512-token repeated-fox prompt
- resident row-128/repacked wide-prefill switches from EXP-0300
- focused layer harness: B512, three control/candidate pairs

## Correctness

The new producer's complete M23-Q8 output was byte-identical to the baseline
SwiGLU plus MMQ-Q8 producer for the existing B128 synthetic GPU fixture. The
focused `qwen35-conv-batch-gpu` test passed. A full P512 candidate run was
finite; no continuation claim is made because it used `--max-tokens 0`.

## Measurements

Layer B512 GPU-event timings, three interleaved pairs:

| path | samples (us) | mean (us) |
|---|---:|---:|
| control | 78358.170, 78290.802, 78274.010 | 78307.661 |
| candidate | 78159.439, 78198.959, 78147.438 | 78168.612 |

The candidate saves `139.05 us/layer`, or about `6.67 ms/P512` if the result
held across all 48 recurrent layers. This is below the 25 ms/P512 harvest
threshold and cannot explain the 2.7 s end-to-end gap.

The single full-model spot-check was 5161.77 ms control versus 5139.87 ms
candidate. It was not treated as a qualified throughput result because this
pair had no continuous telemetry and the process-load time dominates wall
time.

Resident allocation remained `22,801,772,884 B`; no hot-path weight upload was
introduced.

## Decision

**REJECT.** Remove the candidate wiring and retain the byte-identity check in
this record only. Do not revisit this boundary unless a materially different
mapping or a profile showing at least 25 ms/P512 of recoverable work exists.

## Follow-up

Port and measure the pinned mx Q4/Q5/Q6 repacked-MMQ execution contract; this
small producer fusion does not move the primary gate.
