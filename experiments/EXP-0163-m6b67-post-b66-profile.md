# EXP-0163 — M6-B67 post-B66 recurrent profile

## Question

Where does the current production Qwen3.8-27B token spend GPU time after
selecting the dual beta/alpha projection?

## Environment

```text
GPU:       AMD Instinct MI50 / gfx906
Clocks:    profile_peak (1725 MHz SCLK / 1000 MHz MCLK)
Model:     /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Build:     build/mi50-release
Position:  63
```

## Results

```text
total GPU event: 71.6269 ms/token
layer sum:       68.3338 ms/token
final norm:      0.02336 ms
final Q8:        0.00832 ms
LM head:         2.5056 ms
argmax:          0.49664 ms
allocations:     0
```

Representative recurrent-layer stages:

```text
QKV projection:       0.173–0.184 ms
Gate projection:      0.055–0.059 ms
beta/alpha projection: 0.020–0.022 ms
state update:         0.092–0.096 ms
FFN Gate/Up:          0.261–0.277 ms
FFN Down:             0.419–0.446 ms
```

## Interpretation

The dual beta/alpha projection is visible as a lower beta/alpha stage than
the earlier two-GEMV profile, but the repeated recurrent layer remains the
dominant cost. FFN Down is still the largest individual recurrent stage, while
the complete recurrent pipeline—not launch overhead—is the main opportunity.

## Decision

**MEASUREMENT-ONLY.** Keep B66 production-selected. No new isolated kernel
target is selected from this profile without a materially different mechanism.
