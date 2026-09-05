# EXP-0162 — M6-B66 dual beta/alpha FP32 GEMV

## Question

Can the two recurrent beta/alpha projections reuse the same normalized-input
loads without changing their outputs or recurrence contract?

## Candidate

The candidate computes both FP32 output vectors in one 256-thread GEMV. It
keeps the existing weights, input, reduction, outputs, and downstream
`prepare_beta_decay` operation unchanged. The default selects the candidate;
`MIINFER_DUAL_BETA_ALPHA_GEMV=0` selects the two-GEMV control.

## Environment

```text
GPU:       AMD Instinct MI50 / gfx906
Clocks:    stable peak / profile_peak
Model:     /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Build:     build/mi50-release
Fixture:   /tmp/m6a273-reference
```

## Correctness

The candidate passed native 16-token generation replay and the full Release
CTest suite (20/20). Decode allocations remained zero and device usage was
unchanged at 20,094,914,900 bytes.

## Results

| Workload | Control | Candidate | Change |
| --- | ---: | ---: | ---: |
| TG64 | 14.1702 tok/s | 14.2808 tok/s | +0.780% |
| TG128 | 13.9710 tok/s | 14.0302 tok/s | +0.424% |

Control TG64 samples were `4510.29, 4512.44, 4516.52, 4517.73, 4519.86`
ms. Candidate samples were `4470.82, 4480.44, 4481.55, 4484.06, 4484.95`
ms. Candidate TG128 samples were `9107.35, 9115.40, 9123.18, 9126.88,
9133.69` ms; the control median was 9161.80 ms.

## Decision

**KEEP / production-selected.** The candidate is exact under the native
replay gate and removes duplicate normalized-input traversal. The TG128 gain
is below 0.5%, so the improvement is useful but modest.

## Follow-up

Refresh the post-B66 profile before selecting the next optimization. Do not
repeat prior recurrent state-update or FFN Down variants.
