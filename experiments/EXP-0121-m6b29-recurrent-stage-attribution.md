# EXP-0121 — M6-B29 recurrent stage attribution

## Question

Does recurrent layer 0 provide a representative stage profile, or do layers
1 and 2 expose a materially different recurrent cost that should drive the
next optimization?

## Scope

Measurement only. The existing production kernels and defaults were not
changed. The profile harness now records the existing 14 recurrent stage
events for layers 0, 1, and 2 in one P64 run.

## Environment

```text
GPU: AMD Instinct MI50 / gfx906
Clock policy: stable_peak
Observed clocks: 1725 MHz SCLK / 1000 MHz MCLK
Model: /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Build: build/mi50-release
ROCm: 7.1.52802-9999
Fixture: /tmp/m6a273-reference
Command: miinfer-m6a21-qwen35-gpu-hybrid-block MODEL FIXTURE --profile64
```

## Results

The profile's total GPU time was **89.2655 ms/token**. It reported zero
allocations. The 48 recurrent layers account for **64.54 ms** in the previous
repeated profile, versus **21.01 ms** for 16 full-attention layers.

Representative recurrent stage timings at position 63:

| Stage | L0 ms | L1 ms | L2 ms |
| --- | ---: | ---: | ---: |
| attn_norm | 0.02320 | 0.02304 | 0.02336 |
| qkv_projection | 0.17184 | 0.18384 | 0.17328 |
| gate_projection | 0.08368 | 0.08848 | 0.08432 |
| beta_alpha | 0.02960 | 0.03104 | 0.02960 |
| conv_and_head_norm | 0.01504 | 0.01568 | 0.01520 |
| state_update | 0.21296 | 0.22960 | 0.21088 |
| recurrent_gate | 0.01280 | 0.01344 | 0.01392 |
| ssm_output_projection | 0.07120 | 0.07568 | 0.07472 |
| attention_residual | 0.00672 | 0.00704 | 0.00704 |
| ffn_norm | 0.02288 | 0.02400 | 0.02384 |
| ffn_gate_up | 0.42336 | 0.44368 | 0.44608 |
| ffn_activation | 0.00688 | 0.00688 | 0.00736 |
| ffn_down | 0.41840 | 0.41856 | 0.43840 |
| ffn_residual | 0.00848 | 0.00864 | 0.00864 |

Layers 0–2 have the same cost structure. No first-layer-only setup effect or
layer-index-specific anomaly was observed.

## Ranking

| Family | Approximate recurrent cost | Status | Priority |
| --- | ---: | --- | ---: |
| FFN Gate/Up + Down | 0.84–0.88 ms/layer | existing optimized projection paths | 1 |
| State update | 0.21–0.23 ms/layer | not freshly differentially profiled | 2 |
| QKV projection | 0.17–0.18 ms/layer | packed-dot4 path selected | 3 |
| Gate/beta/alpha/conv/output support | ~0.22 ms/layer combined | not separately compared | 4 |

The FFN projection family has already received exact-shape optimization work;
the next candidate must not assume another GEMV geometry win. The largest
uncleared recurrent-specific family is the state update, with an estimated
10–11 ms/token aggregate across 48 layers.

## Decision

**MEASUREMENT-ONLY.** The recurring recurrent profile is now characterized;
production remains B27-selected and the B28 profile remains valid. The next
bounded experiment should inspect or differentially benchmark the recurrent
state-update path, not fuse stages merely to reduce dispatch count.

## Follow-up

Perform one state-update-focused attribution or kernel differential. Preserve
GPU-resident state, zero decode allocations, and the accepted external
observable contract. Require whole-token A/B before production selection.
