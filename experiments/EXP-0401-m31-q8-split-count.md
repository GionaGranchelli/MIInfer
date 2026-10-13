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

On Machinist GPU2 (`0000:88:00.0`, gfx906), the outputs matched exactly
(`max_abs=0`, cosine `1.0`). Five interleaved samples were:

- 32 splits: `168.919, 169.384, 169.492, 169.464, 192.433 ms`; median
  `169.464 ms`.
- 3 splits: `230.354, 204.077, 205.338, 202.968, 202.500 ms`; median
  `204.077 ms`.

Three splits were 20.4% slower. The GPU2 junction temperature peaked at 48 C
in the independent polling samples, below the 80 C warning threshold. Keep 32
splits for this shape; do not change the production dispatch from this result.
