# EXP-0130 — M6-B38 post-B37 production profile

## Question

After rejecting Q4_K weight staging, what does the accepted B35 production
path cost and which distinct Q4_K mechanism remains worth testing?

## Method

The B35 production-default path was profiled at position 63 under stable_peak
after B37 was removed. No production behavior changed.

## Environment

```text
GPU:                    AMD Instinct MI50 / gfx906
Clock:                  stable_peak (1725 MHz SCLK / 1000 MHz MCLK observed)
Model:                  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Build:                  build/mi50-release
Fixture:                /tmp/m6a273-reference-p12
Profile position:       63
```

## Results

```text
Total GPU event:        82.3597 ms/token
Layer sum:              79.1278 ms/token
Final RMSNorm:          0.02416 ms
Final Q8:               0.00864 ms
Final LM head:          2.44496 ms
Final argmax:           0.497759 ms
Allocations:            0
Layers:                 48 recurrent + 16 full-attention
```

Representative recurrent layer-0 stages:

| Stage | GPU ms |
| --- | ---: |
| QKV projection | 0.17376 |
| Gate projection | 0.08384 |
| State update | 0.11408 |
| Recurrent gate | 0.01184 |
| SSM output projection | 0.07584 |
| FFN Gate/Up | 0.35680 |
| FFN activation | 0.00688 |
| FFN Down | 0.41376 |

The representative full-attention layer-3 stages included Q projection
`0.19904 ms`, cached attention `0.14624 ms`, FFN Gate/Up `0.37920 ms`, and FFN
Down `0.43488 ms`.

## Interpretation

The accepted Q4_K×Q8_1 LDS-input path remains the dominant projection path.
B37 showed that staging complete Q4_K blocks adds too much LDS traffic and
reduces throughput. The remaining bounded hypothesis is metadata-only staging:
reuse each block's `d`, `dmin`, and packed scale/min bytes in LDS while leaving
the Q4 nibble loads, per-row reduction, and arithmetic unchanged.

## Decision

**MEASUREMENT-ONLY.** B35 remains production-selected. The next candidate is
metadata-only Q4_K×Q8_1 LDS staging; no additional full-weight staging is
justified.
