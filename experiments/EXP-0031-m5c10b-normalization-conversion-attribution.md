# EXP-0031 — M5-C10b normalization/conversion boundary attribution

**Status:** CLOSED — MEASUREMENT-ONLY
**Milestone:** M5
**Date:** 2026-09-01
**Production policy:** shared Gate/Up Q8 activation reuse
**Model:** Qwen3-8B Q4_0 (Qwen3-8B-q4_0-b968826d.gguf)

## 1. Question

With C9c shared Gate/Up Q8 reuse in production, which normalization,
conversion, and quantization boundaries remain expensive or structurally
avoidable on the real P64 decode path?

## 2. Method

The position audit was extended with measurement-only named boundary stages.
Each stage records deferred HIP-event time, dispatch count, and logical output
bytes. The audit ran the trace-free MI50 path at positions 1, 8, 16, 32, and
64 with GPU argmax and shared Gate/Up Q8 reuse. No production kernel,
precision policy, scheduling policy, or buffer lifetime was changed.

The raw result and environment captures are retained under
bench/results/20260901T-c10b-boundary-clean/. The observed hardware state was
approximately 930/350 MHz SCLK/MCLK, so absolute rates remain low-clock
qualified. Deferred stage totals are attribution evidence, not additive wall
time: they overlap because events are recorded on the active stream.

## 3. P64 production result

| Metric | P64 |
|---|---:|
| Clean production wall | 19.602 ms |
| Whole-token GPU event | 19.733 ms |
| Deferred attributed GPU time | 27.277 ms |
| Dispatches | 1553 |
| Synchronization sites | 38 |
| Temporary allocations | 0 |
| Copy accounting | 589,828 bytes |

The broad post-C9c categories remain:

| Category | GPU ms | Dispatches |
|---|---:|---:|
| FFN projection | 6.815 | 108 |
| Attention | 4.925 | 36 |
| Normalization | 2.876 | 289 |
| Quantization | 3.253 | 433 |
| Conversion | 2.211 | 324 |

## 4. Boundary map

The named stages make the producer / representation / consumer boundaries
explicit:

| Boundary family | P64 GPU ms | Dispatches | Logical bytes | Immediate consumer |
|---|---:|---:|---:|---|
| Attention RMS reduction + norm scale | 0.893 | 72 | 1,179,648 | Q/K/V/O path |
| Q/K head RMS reduction + scale | 1.081 | 144 | 1,179,648 | RoPE / attention |
| FFN RMS reduction + norm scale | 0.888 | 72 | 1,179,648 | Gate/Up/Down path |
| Final RMSNorm | 0.021 | 1 | 16,384 | final Q8_K |
| Attention F32→F16→F32 round-trip | 0.488 | 72 | 884,736 | O projection |
| Projection output F16→F32 | 1.720 | 252 | 5,603,328 | residual / activation |
| Projection input F32→F16 | 1.477 | 216 | 2,359,296 | same projection's Q8 input |
| Projection input Q8 quantization | 1.752 | 216 | 1,327,104 | Q4×Q8 GEMV |
| Final norm→Q8_K | 0.009 | 1 | 4,672 | LM head |

The byte totals are logical output-payload accounting for the named stages,
not a claim about unique physical traffic; the exact per-stage accounting is
in the machine-readable result. The important structural facts are that
shared Gate/Up reuse leaves Up input quantization at zero dispatches, and that
many remaining boundaries are small, repeated 36-times-per-layer operations.

Representative exact P64 stages include:

    ffn_rms_normalize       0.632960 ms / 36 / 589824 bytes
    ffn_norm_scale           0.252000 ms / 36 / 589824 bytes
    q_head_rms_normalize     0.292480 ms / 36 / 589824 bytes
    q_head_scale             0.250880 ms / 36 / 589824 bytes
    attention_f32_to_f16     0.243840 ms / 36 / 294912 bytes
    attention_f16_to_f32     0.243360 ms / 36 / 589824 bytes
    gate_input_f32_to_f16   0.248320 ms / 36 / 294912 bytes
    gate_input_q8            0.288960 ms / 36 / 165888 bytes
    down_input_f32_to_f16   0.248640 ms / 36 / 884736 bytes
    down_input_q8            0.303520 ms / 36 / 497664 bytes

All named stages and all positions are retained in
bench/results/20260901T-c10b-boundary-clean/result.json.

## 5. Interpretation

1. The current path is not allocation-, copy-, or final-logits-transfer
   limited: those are already zero-allocation, 38-sync, and 4-byte-token-ID
   properties.
2. Normalization is genuinely distributed across many small operations.
   FFN normalization is about 0.888 ms of named boundary time, attention
   normalization is about 0.893 ms, and Q/K head normalization is about
   1.081 ms.
3. Conversion is distributed across repeated representation boundaries. The
   attention round-trip is a required semantic boundary and must not be
   removed. Projection output F16→F32 conversion is repeated for every
   projection and remains a candidate only if its consumer contract is
   preserved.
4. The shared Gate/Up policy is visible: Gate input quantization is present,
   Up input quantization has zero dispatches, and both projections consume the
   shared Q8 stream.
5. Since deferred attribution exceeds the 19.735 ms whole-token event, these
   stage times cannot be summed as wall-clock components. C10c must be judged
   by clean end-to-end A/B timing, with dispatch and byte changes as
   supporting evidence.

## 6. Decision

    CLOSED — measurement-only

The next experiment should target one producer/consumer boundary rather than
another standalone GEMV geometry. The selected C10c candidate is an opt-in
FFN RMSNorm + norm-weight scale + F32→F16 + shared Q8_1 path. It must retain
the existing F32 operation and F32→F16 rounding semantics before Q8
quantization wherever that is the current contract. If it cannot preserve the
Q8 stream byte-for-byte, reject it.

## 7. Follow-up

Keep C9c shared reuse as the production default. For C10c, implement only the
selected FFN normalization-to-shared-Q8 candidate after confirming its exact
current producer, representation, and consumer. Require byte-identical Q8,
the existing Release correctness and deterministic 64-token gates, plus clean
P64 A/B timing.
