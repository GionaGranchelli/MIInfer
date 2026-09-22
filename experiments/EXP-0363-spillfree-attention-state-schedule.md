# EXP-0363 — Spill-free attention state schedule

## Hypothesis

Reproducing mx’s state placement—not merely its BQ16/GQA2 geometry—will remove
the EXP-0362 register spills:

* Q in LDS for the kernel lifetime;
* reusable K/V LDS workspace;
* KQ materialized in LDS;
* one compact output accumulator per active row;
* no long-lived per-thread `qv[8]` or `acc[8]` arrays.

## Candidate A compile gate

The isolated candidate compiled for gfx906 with:

| Resource | Result | Gate |
| --- | ---: | ---: |
| VGPR | 40 | no spill |
| VGPR spills | 0 | pass |
| SGPR | 44 | — |
| SGPR spills | 0 | pass |
| Private segment | 0 bytes | pass |
| Dynamic LDS | 28 KiB | pass |
| Workgroup | 256 threads | pass |

This is a substantial resource improvement over EXP-0362’s 137 VGPR spills.

## Runtime status

Timing was stopped before accepting any result. The first P512–P8K check
reported `schedule_max_abs_error=inf`. The prototype currently uses a 32-row
KQ LDS buffer but does not yet make that buffer tile-local across the full KV
sequence. Consequently, longer contexts overwrite earlier KQ positions and the
result is invalid.

Per the experiment gate, no performance conclusion is drawn from those timings.

## Required correction

Make KQ tile-local and carry online max/sum/output state across KV tiles. The
correct lifetime must be:

```text
load Q once into LDS
for each KV tile:
    stage K
    compute tile-local KQ
    update online softmax state
    stage V into the same workspace
    update compact output state
write output
```

The fixed-width causal loop correction made the candidate numerically valid.
The maximum absolute difference versus the control was `2.7563e-05` at every
tested length, within the existing FP16-KV qualification scale but not exact.

The spill-free candidate was then timed at the required lengths:

| Tokens | Control ms | EXP-0360 ms | EXP-0362 ms | EXP-0363 ms | Control / EXP-0363 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 2.588 | 4.576 | 13.897 | 17.925 | 0.144× |
| 2048 | 40.337 | 78.559 | 193.238 | 271.110 | 0.149× |
| 4096 | 176.837 | 304.663 | 716.097 | 1019.530 | 0.173× |
| 8192 | 751.073 | 1221.660 | 2816.900 | 3976.570 | 0.189× |

P16K was not run because the candidate did not win materially at P8K.

## Schedule correction attempt

A tile-local online-softmax v2 was added so KQ no longer aliases the complete
sequence. It was not benchmarked because the mandatory resource gate failed:

| Resource | v2 result |
| --- | ---: |
| VGPR | 64 |
| VGPR spills | 86 |
| SGPR | 56 |
| Private segment | 348 bytes |

The final spill-free v2 resource metadata was 158 VGPRs, 55 SGPRs, zero spills,
zero private bytes, and 25 KiB dynamic LDS. The no-spill gate therefore passed.

However, removing spills did not recover performance. The candidate is slower
than EXP-0362 at every length despite being spill-free. Its implementation
re-stages K for each position and repeats the V staging for each row group; it
also uses a one-resident-workgroup launch. mx instead loads a K tile once,
materializes a compact KQ fragment, and consumes that fragment across all query
rows before reusing the workspace for V. The remaining gap is therefore the
KQ/V dataflow and synchronization schedule, not register spilling.

## Decision

EVIDENCE-BASED REJECTION — the state placement removes register spilling, but
this source-shaped implementation remains substantially slower than the
one-Wave64 control. Do not merge it or change geometry based on this result.
