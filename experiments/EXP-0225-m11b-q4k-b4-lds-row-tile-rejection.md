# EXP-0225 — M11-B Q4_K B=4 Full Row-Tile LDS Rejection

## Hypothesis

The per-K-tile barriers in EXP-0224 could be removed by staging the complete
two-row Q4_K tile set once in LDS. The largest production shape needs only
`2 * 17 * 640 = 21.8 KiB`, leaving room for a 512-thread workgroup.

## Candidate

An opt-in 512-thread kernel staged two complete output rows, synchronized
once, and assigned four token waves to each row. Each wave retained one
accumulator and consumed the staged row without further barriers.

## Correctness

P17 produced the same 16-token greedy continuation as the control.

## Results

Qualification model, MI50 gfx906, native layer-major path, graph disabled:

| Workload | Control | Full-row LDS candidate |
| --- | ---: | ---: |
| P17 | ~45 tok/s | 34.86 tok/s |
| P128 | 47.28 tok/s | 41.78 tok/s |

The one-time staging removed the per-tile barrier cost from EXP-0224, but the
512-thread/two-row mapping still lost to the 256-thread/four-row control.

## Decision

**REJECT.** Remove the temporary kernel. This row-tile shape does not justify
the occupancy and LDS cost on gfx906.
