# EXP-0154 — M6-B62 DeltaNet row-wave state mapping

## Question

Can the transposed DeltaNet state update use one Wave64 per logical state row,
with four rows per 256-thread workgroup, instead of one thread per row?

## Candidate

The opt-in kernel cooperatively loads query/key into LDS. Each Wave64 handles
one logical state row, with two state columns per lane, and uses Wave64 shuffles
for the key and query reductions. The existing beta/decay inputs and state
layout are retained. Enable with `MIINFER_DELTA_ROW_WAVES=1`.

## Environment

```text
GPU:       AMD Instinct MI50 / gfx906
Model:     /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Build:     build/mi50-release
Clock:     profile_peak; 1725 MHz SCLK / 1000 MHz MCLK observed
Fixture:   /tmp/m6a273-reference-p12
Baseline:  B59 expanded Q4_K Down production path
```

## Correctness

The candidate built and remained allocation-free, but it failed quickly. A
16-token run changed the final token from the production result (`585` to
`424`). The external observable run showed large/non-finite recurrent-state
errors beginning at position 2 and multiple wrong teacher-forced decisions.

The failure is not an acceptable floating-point envelope; it is a broken
state-update candidate. No performance result was collected.

## Decision

**REJECT.** Keep the existing one-block-per-head transposed state-update
kernel. Do not revisit this geometry without first proving the state-layout
mapping and reduction contract in a standalone operation test.

## Follow-up

Return to differential profiling and avoid further recurrent geometry changes
until a smaller, independently checkable state-update fixture exists.
