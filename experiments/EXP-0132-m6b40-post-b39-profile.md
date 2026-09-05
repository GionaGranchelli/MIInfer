# EXP-0132 — M6-B40 post-B39 production profile

## Question

What remains after B39's Q4_K×Q8_1 metadata staging, and is the next
metadata experiment still justified?

## Method and environment

The B39 production default was profiled at position 63 on the MI50 under
stable_peak (1725 MHz SCLK / 1000 MHz MCLK), using
`Qwen3.8-27B-Q4_K_M.gguf` and `/tmp/m6a273-reference-p12`.

## Results

```text
Total GPU event:        76.6455 ms/token
Layer sum:              73.3273 ms/token
Final RMSNorm:          0.02320 ms
Final Q8:               0.00832 ms
Final LM head:          2.53280 ms
Final argmax:           0.495839 ms
Allocations:            0
Layers:                 48 recurrent + 16 full-attention
```

Representative layer-0 stages:

| Stage | GPU ms |
| --- | ---: |
| QKV projection | 0.18336 |
| Gate projection | 0.06624 |
| State update | 0.11968 |
| SSM output projection | 0.07504 |
| FFN Gate/Up | 0.31520 |
| FFN activation | 0.00720 |
| FFN Down | 0.42640 |

## Interpretation

B39 reduced Q4_K metadata access cost, but its dot path still reconstructs
the eight scale/minimum pairs repeatedly from packed bytes. FFN Down remains
the largest repeated stage. The next bounded candidate is to decode those
pairs once per block into the existing LDS tile while retaining the same
weight loads, Q8_1 input, dot arithmetic, and reduction.

## Decision

**MEASUREMENT-ONLY.** B39 remains production-selected. Test decoded metadata
reuse as one opt-in candidate; do not change the Q4 weight layout or geometry.
