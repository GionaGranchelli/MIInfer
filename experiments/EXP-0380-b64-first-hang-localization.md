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
| O projection | yes | yes | PASS |
| post-attention norm | yes | yes | PASS |
| FFN gate/up | yes | yes | PASS |
| SwiGLU | yes | yes | PASS |
| FFN down | yes | yes | PASS |
| final output | yes | yes | PASS |

Sparse checkpoints also passed the complete attention path at L7, L11, and
L15. L3, L7, L11, and L15 therefore all completed Q/K/causal plus O/norm/FFN
and final output in the exact composed state.

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
The O, norm, FFN, and final-output probes likewise emitted host-return and
synchronized GPU-event markers.

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
earlier whole-model failure class remains **UNKNOWN / runtime composition
wait**; a GPU kernel noncompletion is not proven.

The follow-up run with explicit `B64 CHUNK_BEGIN`, `L3 PREP_HOST_BEGIN`,
`FINISH_PREFILL_BATCH`, and `FINISH_PREFILL_WIDE ENTER` markers also emitted no
marker before the 120-second stop. This means the O-stage probe can still stop
before reaching L3 in the full prefix; it does not justify attributing the
failure to the O projection itself.

The subsequent run added `GENERIC_B64_CHUNK_BEGIN` before the B64 layer loop
and again emitted no marker before the watchdog stop. That run therefore did
not reach B64 at all. The full P960 prefix is not a stable prerequisite for
the later-stage probe under repeated execution; this is a separate runtime
boundary that must be isolated before interpreting O/FFN stages.

The latest run also added `FULL_CHUNK_BEGIN` at entry to
`prefill_full_layer_major_chunk()` and emitted no marker. It therefore stopped
before the full-layer-major B512/B128 path was entered, during the earlier
P960 prefix. This confirms that the missing O marker is not evidence against
the L3 O projection.

## B128 comparison

Not run as a new probe. Existing EXP-0375 evidence proves the matched B128
route completes. A fresh B128 operation comparison is deferred until the
probe's pre-marker boundary is instrumented more precisely.

## Decision

**LEARN.** No individual B64 attention operation is the first failing
operation. L3, L7, L11, and L15 all pass the complete attention path in the
exact composed state. The earlier whole-model hang therefore depends on
composition outside an individual attention layer.

## Exact next PRIMARY

The next PRIMARY is whole-model B64 inter-layer/resource composition after
individual attention execution: compare the first failing layer handoff,
state/workspace lifetime, and release/swap ordering after the last passing
attention checkpoint. Do not attribute the hang to Q/K/causal/O/FFN kernels,
and do not run B4 or whole-model qualification until that composition contract
is proven.
Do not run FFN, B4, or whole-model qualification.

Is B64 scheduler work authorized? **NO.**

Is B4 work authorized? **NO.**

Experiment SHA: `69f2c9fed6477c8d822e9001585b5521300a7cec` (record commit before provenance amendment).

Graph SHA: `74d94bbb6895394fe22284978e5f035d3a894579` (`graphify-out/graph.json` blob).

Working-tree status: clean after commit and graph refresh.

No production B64 scheduler, residual optimization, or new GPU math kernel was implemented.
