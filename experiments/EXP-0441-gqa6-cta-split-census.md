# EXP-0441 — GQA6 Stage1 CTA and split-work census

**Result:** REJECT split imbalance as the limiting premise.

## Question

Do a few Split-K partitions or query positions receive disproportionate
Stage1 work, making occupancy, barriers, or local K/V reuse the wrong place to
optimize?

## Method

`experiments/tools/gqa6_cta_census.py` reproduces the Stage1 launch's
`ceil(total_length / num_splits)` partitioning and counts key positions for
each `(query token, split)` CTA. GQA6 launches the same CTA geometry for each
of four KV heads and computes six query heads per CTA; those factors multiply
all counts without changing the reported split skew. The cold census covers
absolute query positions 1–65,536; the restored census covers a 73,099-token
prefix followed by 1,024 queries. This is an arithmetic work census, not a
timing or profiler result.

## Results

| Workload | Splits | Active CTAs per KV head | Mean keys / CTA | Range keys / CTA | Max split-work skew |
| --- | ---: | ---: | ---: | ---: | ---: |
| 73,099 + 1,024 restored suffix | 32 | 32,768 | 2,300.4 | 2,255–2,317 | 0.0211% |
| 73,099 + 1,024 restored suffix | 64 | 65,536 | 1,150.2 | 1,081–1,159 | 0.0428% |
| Cold 64K | 32 | 2,095,547 | 1,024.8 | 1–2,048 | 0.0473% |
| Cold 64K | 64 | 4,186,480 | 513.0 | 1–1,024 | 0.0961% |

Every split receives essentially equal total key work. Doubling splits halves
the per-CTA serial scan but doubles the active CTA count, without fixing a
meaningful load imbalance. This agrees with EXP-0423, where 64 splits did not
improve the restored suffix and changed state/logits.

## Decision

Do not test more split counts or try to rebalance split IDs. The remaining
Stage1 problem is the cost of processing the attention interactions, not a
few overloaded split actors. The next design needs a materially different
Q×K execution layout or reduction strategy, with numerical changes stated
before implementation.

Reproduce with:

```bash
python3 experiments/tools/gqa6_cta_census.py
```
