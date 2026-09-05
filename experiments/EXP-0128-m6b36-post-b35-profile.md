# EXP-0128 — M6-B36 post-B35 production profile

## Question

After B35's Q4_K×Q8_1 LDS activation reuse, what is the remaining measured
token cost and which projection family should be investigated next?

## Method

Profiled the production-default native GPU path at position 63 under
stable_peak. No production behavior changed in this milestone.

## Results

```text
GPU:                    AMD Instinct MI50 / gfx906
Clock:                  stable_peak (1725 MHz SCLK / 1000 MHz MCLK observed)
Model:                  Qwen3.8-27B-Q4_K_M.gguf
Profile position:       63
Total GPU event:        82.1223 ms/token
Layer sum:              78.8699 ms/token
Final RMSNorm:          0.02304 ms
Final Q8:               0.00800 ms
Final LM head:          2.46704 ms
Final argmax:           0.49840 ms
Allocations:            0
Layers:                 48 recurrent + 16 full-attention
```

Representative recurrent stages:

| Stage | L0 | L1 | L2 |
| --- | ---: | ---: | ---: |
| QKV projection | 0.164959 | 0.173440 | 0.174079 |
| State update | 0.119680 | 0.114400 | 0.120320 |
| FFN Gate/Up | 0.378080 | 0.356000 | 0.378240 |
| FFN Down | 0.426239 | 0.418720 | 0.435039 |

The representative full-attention layer measured `0.355840 ms` for FFN
Gate/Up, `0.418720 ms` for FFN Down, and `0.138880 ms` for cached attention.

## Interpretation

B35 moved the recurrent Gate/Up stage down substantially, but the long-K Q4_K
FFN Down projection remains about `0.42–0.44 ms` per layer and is now the
largest repeated projection cost. The previous two-row reduction geometry was
rejected for correctness and is not repeated. The next candidate must use a
different Down-specific mechanism and preserve the accepted external contract.

## Decision

**MEASUREMENT-ONLY.** The B35 LDS activation-reuse path remains production
selected; no code or production selection changed in this profile.

## Follow-up

Run one bounded Down-specific Q4_K×Q8_1 kernel experiment, preferably a
memory/instruction mechanism that preserves the existing 128-thread per-row
reduction. Compare whole-token TG64/TG128 medians before considering any
broader geometry change.
