# EXP-0400 — M31 Q8/Q8 GQA head-pair KV reuse

**Status:** IN PROGRESS

**Baseline:** `88fd63eb0cf8b765f37e5f52168c80783df62e7b`

## Objective

Reduce repeated Q8 KV reads in the late-context attention stage without changing
the production CTA grid or the per-query online-softmax result.

## Hypothesis

For Qwen3.8's 24 Q heads and four KV heads, adjacent Q heads in each existing
64-thread CTA belong to the same GQA group. The baseline wave mapping loads the
same four K and V rows once per half-wave. Let half-wave zero load those rows
into a small LDS tile; both half-waves then consume the rows for their distinct
queries. This holds CTA count, per-query work, causal masking, split boundaries,
and stage-two reduction constant.

## Gates

- Compiler: no spills; inspect VGPR/SGPR, LDS, and occupancy for the specialized
  instantiation.
- Correctness: match the existing Q8/Q8 control at `P=8192, S=512`, requiring
  `max_abs <= 0.01` and cosine `>= 0.9999`.
- Performance: five interleaved samples against the same control; keep only if
  median is at least 10% faster. Reject above 3% slowdown.
- Product: no production selector unless the focused gates pass; then profile
  one full-model late-context tile.

## Implementation

A compile-time specialization of the existing Q8/Q8 kernel stages the four
K/V rows for a Q-head pair in LDS. The normal template path and runtime
dispatch remain unchanged. A focused benchmark compares the specialization
with the established Q8/Q8 path.

The first specialization kept the generic four-position temporaries live and
reported 13 VGPR spills. It was rejected before timing. The current version
consumes one staged position at a time to reduce live state; its compiler
resource report is still pending.

## Results

Pending.
