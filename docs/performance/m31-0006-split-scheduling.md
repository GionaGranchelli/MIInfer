# M31-0006 — Split-scheduling experiment

## Hypothesis and method

Production captures a fixed `24 x 64` grid; the device kernel activates a
context-dependent subset. The benchmark A/Bs that same dynamic production
kernel and graph, changing only `MIINFER_ATTENTION_SPLITS` during candidate
capture and replay. Output is compared to the default and an FP64 CPU oracle
(absolute tolerance `2e-3`). No production default or graph strategy was
changed.

## Results

| Context | Default | Candidate | Default graph pair | Candidate graph pair | Candidate result |
|---:|---:|---:|---:|---:|---|
| 8K | 16 | 8 | 0.718 / 0.678 / 0.666 ms | 1.345 / 1.232 / 1.232 ms | consistently slower, ~1.8x |
| 32K | 64 | 32 | 1.039 ms | 1.365 ms | 31% slower, one sample |
| 32K | 64 | 16 | 0.957 ms | 2.580 ms | 2.7x slower, one sample |

All candidate outputs passed numerical and future-KV checks. At 8K the three
adjacent samples make the loss clear. The 32K alternatives each lost by far
more than the 5% screening threshold in their single diagnostic call. No
128K candidate, 64K candidate, or multi-graph strategy was run: the lower
split counts already lose and are not worth promoting to longer tests.

## Decision

Reject reduced split counts as an optimization candidate. Keep the existing
production selector and fixed graph grid. The fixed 64-block launch does not
translate into useful work for every block: active-split count changes with
context, and reducing active work here lengthened each CTA's KV loop enough to
outweigh the smaller reduction. This is an observed result for these shapes,
not proof that all adaptive/multi-graph strategies are inferior.

No production optimization was implemented. Do not spend more GPU time on
split-count sweeps until a different kernel-level hypothesis has evidence.

Raw paired comparisons:
[8K split-8, 3 repeats](../../results/m31-0006-attention-8k-split8-repeats3.log), SHA-256 `69ce1684ef771d59964aa4277738eaa70a1a2b3a3532e8e4dd5a64a9167af5d4`;
[32K split-32](../../results/m31-0006-attention-baseline-32k-schedule-call1.log), SHA-256 `db646e7587af0fd7d3207266c6d618a51cf3799d5cc5d7401edf50a51962754c`;
[32K split-16](../../results/m31-0006-attention-32k-split16-call1.log), SHA-256 `b7f8ee5704288ea614cc83f5b48c26d285260ba511a13af71a10c644c067b0e3`.
