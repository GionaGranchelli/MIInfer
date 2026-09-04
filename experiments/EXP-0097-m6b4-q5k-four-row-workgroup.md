# EXP-0097 — M6-B4 Q5_K four-row workgroup

## Question

Can the Q5_K × Q8_K recurrent output projection use one Wave64 per output row,
with four rows packed into the existing 256-thread workgroup?

## Baseline and candidate

Baseline was the accepted Q5 scale/min-hoisted kernel from `b6a8c49`, followed
by the accepted Q4 metadata-staged kernel from `d1d9fef`. The candidate mapped
four output rows to the four waves of one workgroup and reduced each row over
its own wave. Q5 block arithmetic and scale/min hoisting were otherwise kept.

## Result

The candidate passed the 16-token deterministic replay and resource checks,
but its replay throughput fell from approximately **5.98 tok/s** to **4.72
tok/s** (about **21% slower**). The candidate was not benchmarked further.

## Decision

**REJECT.** The idle-wave pattern is not a useful target for this MI50 Q5_K
kernel. Restore the one-row production mapping.

## Follow-up

Do not retry this geometry family without new hardware evidence.
