# EXP-0159 — M6-B67 transposed versus non-transposed full-model control

## Question

Is the existing non-transposed DeltaNet state path a viable full-model
replacement for the transposed production layout?

## Environment

```text
GPU:       AMD Instinct MI50 / gfx906
Clocks:    profile_peak
Model:     /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Build:     build/mi50-release
Fixture:   /tmp/m6a273-reference-p12
Workload:  TG64
```

## Results

| Path | Median | Throughput | Replay | Allocations |
| --- | ---: | ---: | --- | ---: |
| Non-transposed | 4844.35 ms | 13.2113 tok/s | PASS | 0 |
| Transposed production | 4514.21 ms | 14.1775 tok/s | PASS | 0 |

The non-transposed path is approximately `6.8%` slower. Device usage was
unchanged at `20,094,914,900` bytes.

## Decision

**REJECT as a replacement.** The transposed state layout remains production.
This control does not rule out a future fused reduction kernel, but that
candidate must retain the transposed physical layout and prove its mapping
independently before full-model integration.
