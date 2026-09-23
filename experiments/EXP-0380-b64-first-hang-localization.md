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

The continued sparse ladder passed L19, L23, and L27 with the same complete
tail. The L31 attempt did not enter the probe: ROCm aborted during model
initialization with:

```text
Memory access fault by GPU node-2
Reason: Page not present or supervisor privilege
```

This is hardware/runtime contamination, not evidence against L31. After
telemetry returned to normal, one clean-runtime L31 retry also emitted no
probe marker and timed out before B64; it did not test L31. No later layer was
run.

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

## Extended sparse-ladder evidence

L35 and L39 passed the complete target attention path. A valid 960-token L43
run established this prefix boundary:

```text
B512       base=0   entered and completed
B128 #1    base=512 entered and completed
B128 #2    base=640 entered and completed
B128 #3    base=768 entered, no completion marker before watchdog
B64       base=896 not entered
```

The follow-up with per-layer B128 markers did not reach the prefix, so it does
not identify a B128 layer. The durable attribution is the repeated-B128 chunk
at absolute base 768, not a B64 attention operation.

The added per-layer probe then showed runtime-state variability: one valid
base-768 run completed layers 0–62 and stalled after layer-63 `BEGIN`; a later
bounded run stalled at layer 7 `BEGIN`. Because the first incomplete layer
changes between otherwise equivalent runs, no deterministic layer or attention
operation can be named from this evidence. This is consistent with repeated
B128 workspace/resource/runtime state, not a stable B64 attention-kernel
failure.

The allocation audit added to the same layer markers showed no allocation
churn before a valid run stalled at layer 47 `BEGIN`:

```text
device allocations: 2685
total device bytes: 21,993,243,028
live device bytes: 21,993,243,028
```

Those values were unchanged across every completed base-768 B128 layer. The
failure is therefore not an unexpected per-layer allocation, free, or model
repack operation. The remaining likely class is stream/workspace/driver state,
but no kernel noncompletion is proven.

## Lifecycle follow-up

The test-only probe now emits flushed markers around the repeated base-768
B128 lifecycle: `REC_BEGIN/REC_RETURN`, `PREP_BEGIN/PREP_RETURN`,
`ATTN_BEGIN/ATTN_RETURN`, and `RELEASE_BEGIN/RELEASE_END`. The corrected
binary built successfully. A further bounded P960 attempt did not emit prompt
or chunk-entry output before termination, so it supplies no new operation-level
attribution. The prior valid per-layer and allocation runs remain the evidence
used for the decision below.

## Decision

**LEARN.** No individual B64 attention operation has failed in the qualified
ladder. L3, L7, L11, L15, L19, L23, and L27 all pass the complete attention
path in the exact composed state. L31 remains untested because both attempts
failed before probe entry. The earlier whole-model hang therefore remains a
composition/runtime issue outside the tested individual attention stages.

## Exact next PRIMARY

The next PRIMARY is repeated-B128 stream/workspace/driver-state attribution at
absolute base 768. Allocation churn is falsified by the constant counters;
next inspect explicit stream synchronization, workspace reuse, and driver
wait state across repeated invocations. Do not attribute the hang to a
specific Q/K/causal/O/FFN kernel, and do not run B4 or whole-model
qualification until that runtime contract is proven.
Do not run FFN, B4, or whole-model qualification.

Is B64 scheduler work authorized? **NO.**

Is B4 work authorized? **NO.**

Experiment SHA: `2e08485a5eb85e97210f558df14f21ac60b478b1`.

Graph SHA: `db07567deaad14c424bb410d35faced86b4acc55` (`graphify-out/graph.json` blob).

Working-tree status: clean after commit and graph refresh.

No production B64 scheduler, residual optimization, or new GPU math kernel was implemented.
