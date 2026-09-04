# EXP-0081 — M6-A27.1 Qwen3.8 L54/P1 output attribution

## Question

Where does the first large L54/P1 observable error enter the recurrent layer?

## Method

The common 64-layer executor was run through P1 with the existing production
L54 recurrent layer trace enabled. Only the established operation boundaries
were inspected.

## Results

| Boundary | Max abs | RMS | Relative RMS |
| --- | ---: | ---: | ---: |
| layer input | 1.02168 | 0.109515 | 0.0496428 |
| recurrent state after update | 0.0301082 | 0.000229391 | 0.0409835 |
| attention normalization | 0.447712 | 0.0551093 | 0.0530756 |
| QKV projection | 0.798281 | 0.0665291 | 0.0408106 |
| recurrent output | 0.000902295 | 0.0000481208 | 0.0352869 |
| gated output | 0.887696 | 0.0235961 | 0.0585486 |
| attention residual | 2.27362 | 0.116667 | 0.0494259 |
| post-attention norm | 0.634787 | 0.0549107 | 0.0542816 |
| FFN output | 20.8795 | 0.320547 | 0.157004 |
| layer output | 23.1531 | 0.379572 | 0.100529 |

The L54 input already differs from the external L53 output checkpoint by
`1.02168`. Normalization and QKV retain that difference, while the L54
recurrent output remains close. The gated path is the first substantial
recurrent-layer output discrepancy, and the FFN amplifies it. EXP-0082 traces
the matching L53 output and confirms that L54 inherits the discrepancy from
the preceding layer.

## Decision

**MEASUREMENT-ONLY / A27.1 complete.** No production behavior or tolerance
changed. A27 remains RETEST.

## Follow-up

If needed, trace the L53 gated-path input/projection boundary; keep recurrence
and FFN kernels unchanged.
