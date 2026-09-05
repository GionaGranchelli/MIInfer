# EXP-0143 — M6-B51 post-B50 production profile

## Question

What is the current steady-state cost after rejecting B50 fused recurrent
output, and which production family remains large enough to justify the next
architectural experiment?

## Baseline

The production-selected B41 path with the transposed no-decay-store recurrent
update. B50 is disabled and removed.

## Environment

```text
GPU:       AMD Instinct MI50 / gfx906
Clock:     stable_peak (1725 MHz SCLK / 1000 MHz MCLK observed)
Model:     /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Build:     build/mi50-release
Fixture:   /tmp/m6a273-reference-p12
Command:   miinfer-m6a21-qwen35-gpu-hybrid-block MODEL FIXTURE --profile64
```

## Results

```text
position:              63
total GPU event:       73.8636 ms/token
layer sum:             70.5806 ms/token
final norm:             0.02288 ms
final Q8:               0.00816 ms
final LM head:          2.48720 ms
final argmax:           0.498719 ms
dispatches:             not exposed by native harness
decode allocations:     0
```

Representative production stages:

```text
recurrent FFN Down:     0.41808–0.44592 ms/layer
recurrent QKV:          0.15648–0.18304 ms/layer
recurrent state update: 0.11376–0.11440 ms/layer
attention cached scan:  0.14560 ms (layer 3)
attention Q projection: 0.20912 ms (layer 3)
attention FFN Down:     0.43680 ms (layer 3)
```

## Correctness

The native profile completed with `M6-B2 qwen35 native P64 profile PASS` and
zero decode-loop allocations. B50's native replay and Release CTest evidence
remain recorded separately; no candidate is selected by this profile.

## Interpretation

The post-B50 baseline is unchanged in architecture. Recurrent FFN Down remains
the largest repeated per-layer stage, while the full recurrent/attention layer
mix accounts for nearly all measured layer work. The simple recurrent-output
fusion hypothesis is closed; the next candidate must change a materially
different execution or weight-representation mechanism and must be justified
against this profile.

## Decision

**MEASUREMENT-ONLY.** No production change.

## Follow-up

Evaluate one high-leverage Q4_K FFN weight-access or whole-recurrent execution
mechanism. Do not repeat B48 metadata staging, B49/B50 post-update fusion, or
rejected geometry/split-K variants without new evidence.
