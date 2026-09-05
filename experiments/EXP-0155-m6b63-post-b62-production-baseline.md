# EXP-0155 — M6-B63 post-B62 production baseline

## Question

What is the current production Qwen3.8-27B decode baseline after closing the
B62 DeltaNet row-wave rejection?

## Baseline

The B59 expanded Q4_K Down path, B55 DeltaNet LDS input reuse, transposed
recurrent-state layout, shared Gate/Up Q8 reuse, and GPU greedy argmax remain
enabled. Rejected B61/B62 candidates are disabled.

## Environment

```text
GPU:       AMD Instinct MI50 / gfx906
Clocks:    profile_peak (1725 MHz SCLK / 1000 MHz MCLK policy)
Model:     /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Fixture:   /tmp/m6a273-reference-p12
Build:     build/mi50-release
```

## Results

| Workload | Median | Throughput | Replay | Decode allocations |
| --- | ---: | ---: | --- | ---: |
| TG64 | 4504.03 ms | 14.2095 tok/s | PASS | 0 |
| TG128 | 9171.46 ms | 13.9563 tok/s | PASS | 0 |

Device bytes after setup and peak: `20,094,914,900`.

Raw samples:

```text
TG64: 4496.54, 4501.81, 4504.03, 4509.57, 4512.25 ms
TG128: 9156.75, 9166.74, 9171.46, 9175.32, 9175.66 ms
```

## Decision

**MEASUREMENT-ONLY.** This establishes the post-B62 baseline; it does not
select another optimization.

## Follow-up

Choose the next candidate only from a fresh family-level differential or
profile. Do not revisit the rejected row-wave mapping without a materially
different, independently validated state-layout hypothesis.
