# EXP-0129 — M6-B37 Q4_K×Q8_1 Down weight staging

## Question

Does staging each pair of Q4_K Down-projection output rows in LDS improve the
long-K projection's throughput while retaining the existing 128-thread row
reduction and arithmetic?

## Hypothesis

The Down projection rereads Q4_K metadata and weight blocks for each row. A
256-thread workgroup could stage the two rows' weights in LDS and improve
memory reuse without changing the dot-product or reduction order.

## Baseline and candidate

The baseline is the B35 production path: two independent 128-thread row
reductions sharing an LDS-resident Q8_1 activation tile. The candidate added
LDS staging for both the activation tile and the two rows' Q4_K blocks. It was
opt-in through `MIINFER_Q4K_Q8_1_LDS_WEIGHTS=1`; the default remained the
baseline during validation.

## Environment

```text
GPU:       AMD Instinct MI50 / gfx906
Clock:     stable_peak (1725 MHz SCLK / 1000 MHz MCLK observed)
Model:     /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Build:     build/mi50-release
Fixture:   /tmp/m6a273-reference-p12
Baseline:  B35 production path
Date:      2026-09-05
```

## Correctness

The opt-in candidate passed native 16-token generation and deterministic
replay. Decode allocations remained `0`; tracked and peak device bytes were
`17,019,965,780`.

The complete P64 external observable contract passed without tolerance
changes, including the accepted margin-aware logit contract and poisoned
reset/replay. The run retained the existing deterministic trajectory checks.

## Benchmark

Five-sample medians per process at stable_peak:

| Workload | B35 control median | Candidate median | Change |
| --- | ---: | ---: | ---: |
| TG64 ms | 5145.80 | 5357.85 | +4.12% |
| TG64 tok/s | 12.4373 | 11.9451 | -3.96% |
| TG128 ms | 10445.0 | 10868.0 | +4.05% |
| TG128 tok/s | 12.2547 | 11.7776 | -3.89% |

The candidate preserved the logical projection contract but added dynamic LDS
weight traffic and a larger per-workgroup shared-memory footprint. No
production copy or allocation counters changed.

## Interpretation

The candidate is slower at both generation lengths. The negative result is
repeatable and is not explained by correctness or memory-capacity changes;
staging the Q4_K weights costs more than it saves on this MI50 workload.

## Decision

**REJECT; not production-selected.** Remove the candidate and retain the B35
LDS activation-reuse path.

## Follow-up

Do not repeat weight-staging or additional Down geometry variants without new
profiling evidence. The next optimization must come from a refreshed
post-B37 profile or a different measured family.
