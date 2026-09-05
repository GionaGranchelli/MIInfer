# EXP-0145 — M6-B53 Q4_K×Q8_K versus Q4_K×Q8_1 FFN differential

## Question

Does the existing Q4_K×Q8_K path outperform the production Q4_K×Q8_1 MMVQ
path when used for the Qwen3.8 FFN projections?

## Candidate

Use the existing `MIINFER_Q4K_Q8_1_MMVQ=0` path. This changes only the input
activation representation and selects the already-implemented Q4_K×Q8_K kernel;
it is not a new kernel implementation.

## Environment

```text
GPU:       AMD Instinct MI50 / gfx906
Clock:     stable_peak (1725 MHz SCLK / 1000 MHz MCLK observed)
Model:     /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Build:     build/mi50-release
Fixture:   /tmp/m6a273-reference-p12
```

## Results

Same-build TG64 medians:

| Path | Median ms | Throughput |
| --- | ---: | ---: |
| B41 Q4_K×Q8_1 MMVQ | 4615.04 | 13.8677 tok/s |
| Q4_K×Q8_K fallback | 4721.90 | 13.5539 tok/s |

The existing Q8_K fallback is approximately **2.26% slower**. Native replay
passed and decode allocations/device usage were unchanged.

## Decision

**REJECT.** Keep the B41 Q4_K×Q8_1 path for FFN projections. A
representation-only switch to Q8_K does not provide a performance opportunity.

## Follow-up

Do not revisit this representation switch without a new kernel/access strategy.
The remaining high-upside work must be above this already-tested projection
choice.
