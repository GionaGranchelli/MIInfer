# EXP-0380 — Localize the first B64 attention-composition hang

## Question

Which exact operation first fails to complete when the real P960 execution
reaches B64 attention at `base=896`, `count=64`?

## EXP-0379 evidence

EXP-0379 Candidate A completed recurrent B64 but retained 16 scalar attention
calls. Candidate B enabled batched attention, entered B64, and hung in D state
with GPU utilization 0%. The candidate was reverted.

## Probe design

Added the opt-in test-only selector
`MIINFER_EXP0380_B64_ATTN_PROBE=1`, requiring the existing EXP-0374 scheduler.
The probe uses the real P960 prefix and exact B64 state, enables B64 only in
the probe guards, emits flushed stderr markers, synchronizes at the selected
checkpoint, and exits immediately. No production B64 selector, scheduler
relaxation, kernel, buffer, or math change was retained.

The probe also records the active route selectors from the existing preset:

```text
fp16_kv_cache=1
gqa_tiled_attn_prefill=0
wide_attn_prefill=1
prefill_mx_o=1
prefill_mx_ffn=1
resident_m23_all=1
fp16_prefill_o=0
```

## Target

```text
base=896
count=64
first attention layer=L3
```

## Stage ladder results

| Stage | Host call returned? | GPU synchronized? | Result |
| --- | --- | --- | --- |
| preparation | yes | yes | PASS |
| Q split/norm/RoPE | yes | yes | PASS |
| K norm/RoPE/KV store | no marker | no | watchdog stop |
| causal attention | not attempted | not attempted | STOP |
| O projection | not attempted | not attempted | STOP |
| post-attention norm | not attempted | not attempted | STOP |
| FFN gate/up | not attempted | not attempted | STOP |
| SwiGLU | not attempted | not attempted | STOP |
| FFN down | not attempted | not attempted | STOP |
| final output | not attempted | not attempted | STOP |

Stage 0 emitted:

```text
EXP0380 L3 PREP HOST_RETURN BEGIN
EXP0380 L3 PREP GPU_EVENT END
```

The bounded stage-1 run emitted:

```text
EXP0380 L3 Q_SPLIT_NORM_ROPE HOST_BEGIN
EXP0380 L3 Q_SPLIT_NORM_ROPE HOST_RETURN
EXP0380 L3 Q_SPLIT_NORM_ROPE GPU_EVENT BEGIN
EXP0380 L3 Q_SPLIT_NORM_ROPE GPU_EVENT END
```

The stage-2 run was killed after 120 seconds without a K/KV marker. Since the
marker was not emitted, this run does not prove that the K/KV kernel itself was
entered. No later stage was attempted.

## Host wait evidence

The watchdog killed the stage-2 process before a `/proc/<pid>/stack` capture
was taken. After termination, `rocm-smi` reported:

```text
GPU utilization: 0%
SCLK: 1606 MHz
MCLK: 1000 MHz
temperature: edge 39 C, junction 41 C, memory 39 C
```

Therefore the failure is not explained by thermal or clock throttling. The
failure class remains **UNKNOWN / pre-marker host or HIP-runtime wait**; a GPU
kernel noncompletion is not proven.

## B128 comparison

Not run as a new probe. Existing EXP-0375 evidence proves the matched B128
route completes. A fresh B128 operation comparison is deferred until the
probe's pre-marker boundary is instrumented more precisely.

## Decision

**LEARN / INCOMPLETE LOCALIZATION.** Preparation and Q split/norm/RoPE are
proven to complete in the exact composed state. The first unresolved boundary
is before the K/KV completion marker, but the current evidence is insufficient
to name the exact failing operation.

## Exact next PRIMARY

Add flushed markers around the real B64 chunk entry and the L3 preparation
call-site, plus a bounded `/proc` wchan/stack capture while the process is
still alive. Re-run only the K-stage probe once. The next result must
distinguish a pre-K recurrent/host wait from entry into
`launch_qwen35_fused_k_norm_rope_kv_store_batch_f16`; do not run causal, O, FFN,
B4, or whole-model qualification until that boundary is proven.

Is B64 scheduler work authorized? **NO.**

Is B4 work authorized? **NO.**

Experiment SHA: `9e1259b083e96915344e4992e0d56b6cc35408cd` (record commit before provenance amendment).

Graph SHA: `763f9362fe344c6ae5270464441dade43f8ffa9e` (`graphify-out/graph.json` blob).

Working-tree status: clean after commit and graph refresh.

No production B64 scheduler, residual optimization, or new GPU math kernel was implemented.
