# EXP-0223 — M11-B Q4_K B=4 Two-Wave Row Rejection

## Hypothesis

Splitting each B=4 output row across two Wave64s over disjoint K tiles could
hide quantized projection latency while preserving the validated B=4 batch
size and avoiding the rejected B=8 register mapping.

## Candidate

An opt-in 512-thread kernel assigned two waves per output row, reduced the
partial sums through LDS, and covered the existing Q4_K tile counts. The
qualified B=4 kernel remained the control.

## Correctness

The candidate produced the exact same P17 greedy continuation as the control:
`你好`, followed by the same 16-token response. The kernel host tests also
passed.

## Results

Same production environment and native layer-major settings as EXP-0221;
prefill-only measurements used `--max-tokens 0`.

| Workload | Control | Two-wave candidate |
| --- | ---: | ---: |
| P128 | 47.28 tok/s | 47.00 tok/s |
| P513 | median 45.90 tok/s | 45.60 tok/s |

Candidate P513 runs were `45.60 / 45.65 / 45.55 tok/s`.

## Decision

**REJECT.** The two-wave row split is neutral to slightly slower in the
production shapes. The temporary kernel and opt-in dispatch were removed.
The remaining gap still requires a materially different quantized skinny-GEMM
dataflow.
