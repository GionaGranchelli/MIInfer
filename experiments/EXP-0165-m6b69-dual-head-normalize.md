# EXP-0165 — M6-B69 dual query/key head normalization

## Question

Can recurrent query and key head L2 normalization share one dual-output
kernel without changing the validated result or harming throughput?

## Candidate

An opt-in 128-thread kernel computes query and key norms together, preserving
the existing per-vector reduction and normalization formula. Enable with
`MIINFER_DUAL_HEAD_NORMALIZE=1`.

## Environment

```text
GPU:       AMD Instinct MI50 / gfx906
Clocks:    stable peak / profile_peak
Model:     /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Build:     build/mi50-release
Fixture:   /tmp/m6a273-reference
```

## Correctness

Native 64-token replay passed. Decode allocations remained zero and device
usage remained 20,094,914,900 bytes.

## Results

| Workload | B66 control | Candidate | Change |
| --- | ---: | ---: | ---: |
| TG64 | 14.2808 tok/s | 14.3157 tok/s | +0.24% |

Candidate samples were `4456.16, 4464.03, 4470.63, 4471.27, 4471.54`
ms. The result is below the useful end-to-end threshold.

## Decision

**REJECT.** The candidate remains diagnostic-only; separate query/key
normalization is restored as the production path.
