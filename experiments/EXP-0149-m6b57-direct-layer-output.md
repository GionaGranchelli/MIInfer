# EXP-0149 — M6-B57 direct layer output

## Question

Can the native Qwen3.8 executor write each layer residual directly into the
caller's output buffer instead of writing `layer_output` and copying 16 KiB?

## Hypothesis

Removing one 16 KiB device-to-device copy per layer removes 64 copies and
1,048,576 bytes of traffic per generated token.

## Candidate

The opt-in `MIINFER_DIRECT_LAYER_OUTPUT=1` path writes the final residual
directly to `output`. The default retains the scratch-buffer plus D2D-copy
path. No arithmetic, layer ordering, state layout, or kernel geometry changed.

## Environment

```text
GPU:       AMD Instinct MI50 / gfx906
Model:     /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Build:     build/mi50-release
Clock:     stable_peak; 1725 MHz SCLK / 1000 MHz MCLK observed
Fixture:   /tmp/m6a273-reference-p12
```

## Correctness

Both control and candidate passed native 64-token generation replay. The last
token and state fingerprint were identical, decode allocations remained zero,
and tracked/peak device bytes remained `17,019,965,780`. Release CTest was
not changed by the candidate and remains 20/20.

## Results

Same-build five-sample medians:

| Workload | Control tok/s | Candidate tok/s | Change |
| --- | ---: | ---: | ---: |
| TG64 | 14.1004 | 14.1039 | +0.02% |
| TG128 | 13.8573 | 13.8449 | -0.09% |

The candidate removes the intended per-layer output copy, but the end-to-end
result is neutral within run variation and slightly negative at TG128.

## Interpretation

The native path's 16 KiB layer-output copies are not a measurable throughput
bottleneck at the current MI50 workload. This confirms the earlier M5 result
for the corresponding copy family in the native executor.

## Decision

**REJECT.** Keep the existing scratch/output path and do not select the direct
output candidate.

## Follow-up

Do not revisit this copy removal without evidence that the execution or clock
regime has changed materially. The next candidate must reduce measured GPU
work rather than only remove small D2D traffic.
