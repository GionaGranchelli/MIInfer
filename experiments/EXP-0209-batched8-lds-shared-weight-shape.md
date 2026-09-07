# EXP-0209 — B=8 LDS Shared-Weight Shape Check

## Hypothesis

Processing two B=4 accumulator groups after one LDS weight load would extend
the validated B=4 Q4_K kernel to B=8 without the register pressure of the
earlier multi-wave mapping.

## Candidate

Temporary Q4_K gfx906 kernel:

- four output waves per workgroup;
- one shared LDS copy of four output rows;
- two sequential four-token accumulator groups;
- corrected `(rows + 3) / 4` grid mapping.

## Correctness

The B=8 output was bit-identical to two existing B=4 launches on a synthetic
64-row Q4_K workload. This isolated kernel check passed.

## Results

| Shape | B=8 LDS | Two B=4 | Relative |
| --- | ---: | ---: | ---: |
| 64 x 1024 | 0.00688 ms | 0.00855 ms | 1.24x |
| 64 x 5120 | 0.05990 ms | 0.01564 ms | 0.26x |

The 1024-column win is not representative of this model's dominant hidden
dimension and FFN projections. On the production-relevant 5120-column shape,
LDS staging and the extra kernel mapping cost roughly 3.8x the B=4 pair.

## Decision

**REJECT.** The kernel and temporary benchmark were removed. The result is a
shape-dependent negative: a synthetic narrow GEMV win does not justify a
production B=8 path for Qwen3.8-27B.

## Follow-up

Only revisit B>4 with a production-shape benchmark and a mapping that reduces
the dominant 5120/17408-column cost, not a narrow-vector microbenchmark.
