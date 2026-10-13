# EXP-0401 — M31 Q8/Q8 Split-K count

**Status:** IN PROGRESS

**Baseline:** `88fd63eb0cf8b765f37e5f52168c80783df62e7b`

## Objective

Check whether the current 32-way Q8 attention reduction is over-partitioned at
`P=8192, S=512`. The experiment changes only the split count, comparing the
current 32 splits with the 3-split setting used by the prior long-prefix GQA
route.

## Gates

- Compare Q8/Q8 outputs from both split counts (`max_abs <= 0.01`, cosine
  `>= 0.9999`).
- Use five interleaved timing samples on the same MI50, exact operands and
  `P=8192, S=512` shape.
- If 3 splits are at least 10% faster, route only the supported long-query
  workload through the lower split count, then profile one full-model tile.
- If slower, retain 32 and move to a different bottleneck.

## Result

Pending.
