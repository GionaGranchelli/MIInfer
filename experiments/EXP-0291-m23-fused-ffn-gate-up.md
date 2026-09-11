# EXP-0291 — Fused wide FFN Gate/Up MMQ

## Hypothesis

The dominant recurrent FFN Gate/Up family can share activation loads and fuse
the SiLU epilogue when both Q4 projections are evaluated by one workgroup.

## Candidate

An opt-in row-128 Q4 MMQ kernel expanded Gate and Up tiles together and wrote
the SiLU product directly to the activation buffer.

## Correctness

The release build succeeded, but the existing model-sized M23 validation
rejected the candidate:

```text
output_max_abs=5.0089
qkv_max_abs=3.8147e-05
gate_max_abs=6.67572e-06
state_max_abs=2.28882e-05
history_max_abs=2.28882e-05
```

The large output error means the fused arithmetic/layout path is not a valid
optimization, regardless of its unmeasured speed potential.

## Decision

**REJECT.** The experimental kernel, API, and switch were removed. The two
validated MMQ projections plus the separate SiLU kernel remain active.
