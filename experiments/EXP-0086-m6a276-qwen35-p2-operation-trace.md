# EXP-0086 — M6-A27.6 Qwen3.8 P2 L0–L2 operation trace

## Question

Which operation boundary first turns the small P2 activation drift into the
observable low-margin token mismatch?

## Method

Added one diagnostic mode to the existing 64-layer GPU harness. It attaches
the established `RecurrentTrace` hooks to L0, L1, and L2 and runs only P0–P2.
The trace compares the recurrent state after P1 and the major L0–L2
boundaries at P2. No production behavior, tolerance, or arithmetic changed.

```bash
cmake --build build/mi50-release \
  --target miinfer-m6a21-qwen35-gpu-hybrid-block -j2
build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  /tmp/m6a273-reference --prefix64-l0-l2-p2-trace
```

## Results

All values below are max absolute error against the external P2 checkpoint.

| Boundary | L0 | L1 | L2 |
| --- | ---: | ---: | ---: |
| State after P1 / state at P2 | 1.90735e-6 | 0.00111466 | 0.00384565 |
| Attention RMSNorm | 0 | 0.00229096 | 0.0128480 |
| QKV projection | 3.81470e-6 | 0.0172018 | 0.0453629 |
| Recurrent output | 4.76837e-7 | 4.25156e-5 | 7.81086e-5 |
| Gated output | 4.76837e-7 | 0.00821006 | 0.00629032 |
| Attention residual | 0.00164509 | 0.0216007 | 0.0276203 |
| Post-attention norm | 0.000302315 | 0.00631595 | 0.0163949 |
| FFN output | 0.00111890 | 0.00405318 | 0.0181277 |
| Layer output | 0.000524521 | 0.0256538 | 0.0457478 |

The detailed RMS/relative-RMS measurements were:

| Layer | Boundary | RMS | Relative RMS |
| ---: | --- | ---: | ---: |
| L0 | attention residual | 2.38361e-5 | 0.00011407 |
| L0 | layer output | 0.000134354 | 0.000572944 |
| L1 | QKV projection | 0.00369514 | 0.00812005 |
| L1 | attention residual | 0.000683617 | 0.00232100 |
| L1 | layer output | 0.00100365 | 0.00329474 |
| L2 | QKV projection | 0.00916865 | 0.0186283 |
| L2 | attention residual | 0.00129842 | 0.00381148 |
| L2 | layer output | 0.00158644 | 0.00500848 |

## Interpretation

The first measurable discrepancy is already present at L0 input/operation
boundaries, but the L0 recurrent update and gated output remain at or near
roundoff. L0 attention/residual and FFN support work introduce the first
visible layer-output error. L1 then shows a larger representation difference
at QKV (`0.0172018`) and L2 reaches `0.0453629` at QKV, while recurrent
outputs themselves remain much closer than the surrounding activations.

This is a distributed early precision/representation drift, not a sudden
L53/L54 implementation failure. It is consistent with the P2 final logits
being highly aligned but changing a decision with reference margin only
`0.0349064`.

## Decision

**MEASUREMENT-ONLY / A27.6 complete.** The first concrete investigation target
is now the L0–L2 attention/residual and QKV representation boundary. Do not
change tolerances, select production behavior, or resume L53/L54 debugging.

## Follow-up

If strict external argmax agreement remains mandatory, trace the L0 attention
residual and L1/L2 QKV input/quantization contracts next. Prefer a single
operation-level comparison or replay over another whole-model scan.
