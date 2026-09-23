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
| K norm/RoPE/KV store | yes | yes | PASS |
| causal attention | yes | yes | PASS |
| O projection | no O marker | no | bounded timeout before O boundary |
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

The bounded K-stage rerun emitted:

```text
EXP0380 L3 K_NORM_ROPE_KV HOST_BEGIN
EXP0380 L3 K_NORM_ROPE_KV HOST_RETURN
EXP0380 L3 K_NORM_ROPE_KV GPU_EVENT BEGIN
EXP0380 L3 K_NORM_ROPE_KV GPU_EVENT END
```

The bounded causal rerun emitted matching host-return and GPU-event markers.
The O-stage run did not emit any EXP-0380 marker and was stopped after 120
seconds, so it does not prove that O projection was entered.

## Host wait evidence

The O-stage watchdog sampled the live process repeatedly. Its `wchan` samples
were mostly `0`, with isolated `folio_wait_bit_common` and `futex_do_wait`
observations. Reading `/proc/<pid>/stack` returned `Permission denied` before
the process was killed, so no kernel stack was captured. After termination,
`rocm-smi` reported:

```text
GPU utilization: 0%
SCLK: 1606 MHz
MCLK: 1000 MHz
temperature: edge 39 C, junction 41 C, memory 39 C
```

Therefore the failure is not explained by thermal or clock throttling. The
failure class remains **UNKNOWN / pre-O host or HIP-runtime wait**; a GPU
kernel noncompletion is not proven.

The follow-up run with explicit `B64 CHUNK_BEGIN`, `L3 PREP_HOST_BEGIN`,
`FINISH_PREFILL_BATCH`, and `FINISH_PREFILL_WIDE ENTER` markers also emitted no
marker before the 120-second stop. This means the O-stage probe can still stop
before reaching L3 in the full prefix; it does not justify attributing the
failure to the O projection itself.

## B128 comparison

Not run as a new probe. Existing EXP-0375 evidence proves the matched B128
route completes. A fresh B128 operation comparison is deferred until the
probe's pre-marker boundary is instrumented more precisely.

## Decision

**LEARN / INCOMPLETE LOCALIZATION.** Preparation, Q split/norm/RoPE, K/RoPE/KV,
and causal attention are proven to complete in the exact composed state. The
first unresolved boundary is after L3 causal attention and before the O marker;
the current evidence is insufficient to name O projection itself as the
failing operation.

## Exact next PRIMARY

Add a flushed marker immediately on entry to `finish_prefill_wide()` and around
the existing O host-call boundary. Capture a user-space debugger/backtrace if
the process remains in `wchan=0`; `/proc/<pid>/stack` is unavailable under the
current permissions. Do not run FFN, B4, or whole-model qualification until
the post-causal/pre-O boundary is proven.

Is B64 scheduler work authorized? **NO.**

Is B4 work authorized? **NO.**

Experiment SHA: `9e1259b083e96915344e4992e0d56b6cc35408cd` (record commit before provenance amendment).

Graph SHA: `763f9362fe344c6ae5270464441dade43f8ffa9e` (`graphify-out/graph.json` blob).

Working-tree status: clean after commit and graph refresh.

No production B64 scheduler, residual optimization, or new GPU math kernel was implemented.
