# EXP-0164 — M6-B68 fused beta/alpha preparation

## Question

Can the selected dual beta/alpha projection also perform the existing
beta/decay preparation in the same kernel without changing results?

## Candidate

The opt-in kernel computes both FP32 projections, then applies the existing
sigmoid/softplus/exp formulas and writes the existing `beta` and `decay`
buffers. The control is the production-selected B66 dual projection followed
by the separate preparation kernel.

Enable with `MIINFER_DUAL_BETA_ALPHA_PREPARE=1`.

## Environment

```text
GPU:       AMD Instinct MI50 / gfx906
Clocks:    stable peak / profile_peak
Model:     /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Build:     build/mi50-release
Fixture:   /tmp/m6a273-reference
```

## Correctness

Native 16-token generation replay passed. Decode allocations remained zero and
device usage remained 20,094,914,900 bytes.

## Results

| Workload | B66 control | Candidate | Change |
| --- | ---: | ---: | ---: |
| TG64 | 14.2808 tok/s | 14.3007 tok/s | +0.14% |

Candidate TG64 samples were `4467.54, 4470.03, 4475.31, 4475.69, 4480.40`
ms. Replay passed.

## Decision

**REJECT.** The candidate removes one small preparation boundary but produces
no useful end-to-end gain. The B66 dual beta/alpha projection remains the
production default; this path remains diagnostic-only.
