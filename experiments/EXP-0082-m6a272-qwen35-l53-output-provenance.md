# EXP-0082 — M6-A27.2 Qwen3.8 L53/P1 output provenance

## Question

Does the L53 output already contain the discrepancy that appears as the L54
input at P1?

## Method

The common 64-layer GPU executor was run with a narrow L53/P1 path capture.
The existing production L53 recurrent layer was instrumented at its established
boundaries. No production kernel, precision contract, or tolerance was changed.

```bash
cmake --build build/mi50-release \
  --target miinfer-m6a21-qwen35-gpu-hybrid-block -j2
build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  /tmp/m6a264-reference.YQO1ad --prefix64-l53-attribution
```

## Results

| Boundary | Max abs | RMS | Relative RMS |
| --- | ---: | ---: | ---: |
| L53 layer input | 0.506977 | 0.0702709 | 0.0372086 |
| attention normalization | 0.340849 | 0.0416377 | 0.0388678 |
| QKV projection | 0.916768 | 0.0604035 | 0.0323530 |
| recurrent output | 0.00111498 | 0.0000677049 | 0.0244824 |
| gated output | 4.07845 | 0.0542933 | 0.125828 |
| attention residual | 1.17761 | 0.101169 | 0.0485735 |
| post-attention norm | 0.334267 | 0.0517101 | 0.0556106 |
| FFN output | 0.155970 | 0.0396576 | 0.0744691 |
| L53 layer output | 1.02168 | 0.109515 | 0.0496428 |

The L53 layer output matches the L54 input discrepancy observed in EXP-0081.
L53 recurrent output remains close, while the gated path is the first large
internal recurrent-layer discrepancy. The L54 mismatch is therefore inherited
from L53 output; it is not first introduced by L54 state storage or recurrence.

## Decision

**MEASUREMENT-ONLY / A27.2 complete.** No production behavior or tolerance
changed. A27 remains RETEST.

## Follow-up

Do not reopen recurrent state mechanics. If another diagnostic is justified,
trace the L53 gated-path input/projection boundary or adjudicate the external
contract for that path. Otherwise proceed with the bounded decision on whether
the observed activation difference is acceptable under the external reference
envelope.
