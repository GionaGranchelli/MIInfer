# EXP-0224 — M11-B Q4_K B=4 LDS Token Distribution Rejection

## Hypothesis

The B=8 register-pressure failure could be avoided by distributing four
prompt tokens across separate Wave64s, staging each pair of output-row Q4_K
tiles once in LDS, and retaining one accumulator per wave.

## Candidate

An opt-in 512-thread, two-output-row kernel staged one native Q4_K tile per
row and synchronized once per K tile. Each wave computed one token for one
output row.

## Correctness

P17 produced the same 16-token greedy continuation as the control. The
candidate was therefore screened on numerical/runtime correctness before
performance classification.

## Results

Qualification model, MI50 gfx906, native layer-major path, graph disabled:

| Workload | Control | LDS-token candidate |
| --- | ---: | ---: |
| P17 | ~45 tok/s | 39.33 tok/s |
| P128 | 47.28 tok/s | 39.59 tok/s |

The candidate added LDS staging and a barrier per K tile, while using a
512-thread workgroup for only two output rows. The overhead outweighed the
reduced per-wave accumulator state.

## Decision

**REJECT.** The temporary kernel and dispatch were removed. A larger-batch
candidate needs tile reuse without a barrier on every quantized K tile and
without reducing row-level occupancy this far.
