# EXP-0125 — M6-B33 post-B32 production profile

## Question

After B32 removes the transposed recurrent decayed-state store, where does the
remaining Qwen3.8-27B P64 token cost go, and what should be measured next?

## Method

Profiled the production-default native GPU path at position 63 under the
existing stable_peak policy. No production behavior changed in this milestone.

## Environment

```text
GPU:       AMD Instinct MI50 / gfx906
Clock:     stable_peak (1725 MHz SCLK / 1000 MHz MCLK observed)
Model:     /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Build:     build/mi50-release
Fixture:   /tmp/m6a273-reference-p12
Production: transposed state + no-decay-store
```

## Results

Two post-B32 profile runs measured `84.6009` and `84.6991 ms` total GPU time
per token. Layer sums were `81.362` and `81.4426 ms`. Final operations in the
second run were:

| Stage | GPU ms |
| --- | ---: |
| Final RMSNorm | 0.0232 |
| Final Q8 materialization | 0.00832 |
| LM head | 2.47872 |
| GPU argmax | 0.490079 |

The executor contains 48 recurrent and 16 full-attention layers. Representative
position-63 stage timings were:

| Stage | L0 | L1 | L2 | Full-attention L3 |
| --- | ---: | ---: | ---: | ---: |
| QKV / Q projection | 0.173599 | 0.18304 | 0.1736 | 0.1992 |
| State update / cached attention | 0.11296 | 0.12256 | 0.1136 | 0.13792 |
| FFN Gate/Up | 0.424 | 0.44672 | 0.445919 | 0.42368 |
| FFN Down | 0.418879 | 0.41408 | 0.43984 | 0.418399 |

The recurring FFN projection pair therefore costs approximately `0.84–0.89
ms` per layer across 48 recurrent layers. The full-attention layer profile
also remains dominated by its FFN pair, while cached attention itself is
`0.13792 ms` at this position. Profiled allocations were `0`; the accepted
production path retains GPU-resident state and workspace.

The immediately preceding clean production medians are:

| Workload | Median ms | Median tok/s |
| --- | ---: | ---: |
| TG64 | 5305.16 | 12.0637 |
| TG128 | 10757.8 | 11.8983 |

## Interpretation

B32 successfully moved state-update work down the ranking. The remaining
profile is real device work rather than an allocation, copy, or synchronization
growth problem. Recurrent FFN Gate/Up and Down projections are the largest
repeated family, but the prior two-row geometry experiment was correctness
unsafe and is not repeated here.

## Decision

**MEASUREMENT-ONLY.** No kernel or production-selection change was made.

The next experiment should target one new, evidence-backed recurrent FFN
projection mechanism and must be evaluated against the accepted external
observable contract and whole-token TG64/TG128 medians.

## Follow-up

Before implementing another geometry variant, characterize whether a new
weight/input reuse or memory-traffic mechanism can reduce the recurrent FFN
projection pair without changing its quantized arithmetic contract.
