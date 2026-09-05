# EXP-0134 — M6-B42 post-B41 production profile

## Question

What does the accepted Q4_K×Q8_1 decoded-metadata path cost after B41, and
which fixed per-token stages remain large enough to justify another bounded
experiment?

## Environment

```text
GPU:       AMD Instinct MI50 / gfx906
Clock:     stable_peak (1725 MHz SCLK / 1000 MHz MCLK observed)
Model:     /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Build:     build/mi50-release
Fixture:   /tmp/m6a273-reference-p12
Position:  63
```

## Results

```text
Total GPU event:  73.8215 ms/token
Layer sum:        70.5777 ms/token
Final RMSNorm:     0.02432 ms
Final Q8:          0.00848 ms
Final LM head:     2.45504 ms
Final argmax:      0.49696 ms
Allocations:       0
Layers:            48 recurrent + 16 full-attention
```

Representative stages were:

| Stage | L0 recurrent | L3 full attention |
| --- | ---: | ---: |
| QKV / Q projection | 0.15632 ms | 0.19936 ms |
| State / cached attention | 0.11328 ms | 0.13824 ms |
| FFN Gate/Up | 0.26000 ms | 0.26064 ms |
| FFN Down | 0.41792 ms | 0.41840 ms |

The profile is lower than B40's `76.6455 ms` total event, but event-category
timings are deferred and noisy; the clean production benchmark remains the
performance authority.

## Decision

**MEASUREMENT-ONLY.** B41 remains production-selected. The largest repeated
family remains Q4_K FFN Down, but no new geometry candidate is selected from
this profile alone. The isolated Q6_K×Q8_1 LM-head metadata candidate is
recorded separately in EXP-0135.

## Follow-up

Continue with one measured whole-token opportunity; do not repeat rejected
full-weight staging or arbitrary workgroup geometry variants.
