# EXP-0168 — post-B1 whole-token profile

## Question

What dominates the current native Qwen3.8-27B decode token after B66?

## Environment

MI50/gfx906, stable-peak setup, Q4_K_M model, `build/mi50-release`, position
63, current production path. Same model and fixture as EXP-0167.

## Results

```text
total GPU event: 71.9133 ms/token
layer sum:       68.6269 ms/token
final norm:       0.02432 ms
final Q8:         0.00848 ms
LM head:          2.50016 ms
argmax:           0.493599 ms
allocations:      0
```

Representative recurrent stages (ms): QKV `0.154–0.184`, gate
`0.055–0.061`, beta/alpha `0.021`, state update `0.091–0.094`, FFN Gate/Up
`0.261–0.278`, FFN Down `0.419–0.446`, and SSM output projection `0.072–0.076`.

## Interpretation

GPU execution remains the dominant cost. FFN Down is the largest repeated
individual stage, but its expansion, metadata, staging, MMVQ, LDS, split-K,
and geometry families have already been tested. A materially different
whole-pipeline or representation hypothesis is required.

## Decision

**MEASUREMENT-ONLY.** No new implementation candidate is selected from this
profile alone.
