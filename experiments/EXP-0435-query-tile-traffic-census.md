# EXP-0435 — Query-tile reuse census after long-context screens

**Status:** CROSS-TOKEN COOPERATIVE CTA TILING REJECTED; ROOT LIMITER STILL OPEN

## Question

Did per-query Split-K boundaries prevent cross-token K/V reuse in the exact
GQA6 query-tile candidates, or did those candidates already share nearly all
the theoretically reusable K/V rows?

## Census

`tools/gqa6_query_tile_census.py` applies the current kernel's split rule to a
restored prefix of 73,099 tokens, 1,024 suffix queries and 32 Split-K splits.
For each query group and split it counts the union of the original per-query
K ranges that a cooperative CTA would need to load. It reports rows for one
KV head; the real kernel has four KV heads. The census counts logical K/V rows,
not physical HBM transactions, cache hits, vector instructions or elapsed time.

| Query tile | CTAs per KV head | Rows if loaded per query | Rows staged as CTA union | Ideal load reuse |
|---:|---:|---:|---:|---:|
| 1 | 32,768 | 75,378,176 | 75,378,176 | 1.00x |
| 2 | 16,384 | 75,378,176 | 37,705,216 | 2.00x |
| 4 | 8,192 | 75,378,176 | 18,860,800 | 4.00x |
| 8 | 4,096 | 75,378,176 | 9,438,592 | 7.99x |

The tile-2 result corresponds to the pair-token kernel's union interval. The
tile-4/8 rows model the same load-once/consume-per-query idea. Split-boundary
variation adds only a small amount over the ideal reductions in this workload;
it does not explain the measured slowdowns.

## Evidence from real-model runs

- EXP-0425 tile-2 was exact and improved the 4K restored suffix by only
  0.7–1.0% across three pairs per card.
- EXP-0427 tile-4 and tile-8 were exact, had zero spills, and were respectively
  61% and 6.6% slower in the measured 4K screens.
- EXP-0413's earlier tile-2/4 variants were 19% and 51% slower; those used an
  older, unpacked control and are supporting evidence only.

## Interpretation

The boundary premise is falsified for this late-context workload: almost all
K/V positions can be shared while retaining each query's split bounds. But
the real-model runs show that reduced logical K/V loads alone do not improve
latency. EXP-0427 rules out register spilling as the sole explanation. Likely
costs include duplicated per-query QK/softmax/value arithmetic, instruction
issue, synchronization and changed CTA parallelism; this census cannot
distinguish them. Do not test another cooperative query-tile size.

## Next action

Collect hardware counters and code-object evidence for the production packed
GQA6 Stage1 on the restored long-context workload. Determine whether the
limiting resource is HBM/L2 traffic, vector issue/softmax work, occupancy, or
LDS synchronization before choosing the next kernel change. Keep exact GQA6
as the control and the original production Q8 path as fallback.
