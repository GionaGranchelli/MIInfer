# EXP-0160 — M6-B68 DeltaNet ordered row-wave reduction

## Question

Can the rejected four-row transposed DeltaNet mapping be made viable by
replacing Wave64 tree reductions with ordered shared partial accumulation?

## Candidate

The opt-in B62 row-wave mapping was retained: four independent state rows per
256-thread workgroup, two columns per lane, and the existing transposed state
layout. Key and query dot products were reduced by storing 64 lane partials in
shared memory and accumulating them in lane order. Production remained
disabled.

## Results

The candidate built, replayed deterministically, and retained zero decode
allocations. It nevertheless failed the external contract at position 0,
layer 58:

```text
max_abs:        2.93093
RMS:             0.153565
relative RMS:    0.0545239
```

The 16-token endpoint also changed from the production result (`585` to
`46194`). No performance benchmark was collected.

## Decision

**REJECT.** Ordered partial accumulation did not make the row-wave mapping
reference-correct. Restore and retain the existing one-block-per-head
transposed state-update kernel.
