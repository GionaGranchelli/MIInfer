# EXP-0233 — M11-B quantized projection Amdahl ceiling

## Question

Can the remaining M11-B gap be closed by improving the existing quantized
projection family alone?

## Baseline

The qualified production-shaped layer-major result is `45.56 tok/s` at P513
with the current native Q4/Q5/Q6 B=4 mappings and deferred tails.

## Evidence

EXP-0230 measured a raw int8 `M=32,N=5120,K=17408` GEMM at `1.38704×` the
native Q4_K B=4 control. That is an optimistic ceiling: it excludes Q4_K
groupwise scale/minimum handling, repacking, and quantized activation costs.

EXP-0231 and EXP-0232 measured proper repacked Q4_K MMQ candidates at 64 and
16 tokens. The 64-token tile reached `1.125×` only at a full tile and the
16-token tile reached `0.351×` the four-launch B=4 control.

## Amdahl calculation

Applying the raw-int8 ceiling to every unit of the current end-to-end path
would produce only:

```text
45.56 tok/s × 1.38704 = 63.19 tok/s
```

That is an intentionally generous upper bound and remains below the 100
tok/s gate. In the representative recurrent profile from EXP-0210, measured
quantized projections account for `1.017 ms` of `1.259 ms` (`80.78%`) of the
layer event. Applying the same ceiling only to that measured share gives:

```text
speedup = 1 / (1 - 0.8078 + 0.8078 / 1.38704) = 1.291×
45.56 tok/s × 1.291 = 58.82 tok/s
```

The layer profile is representative rather than a full-model proof, so the
first calculation is the conservative branch-level bound and the second is
the workload-shaped estimate.

## Interpretation

No local Q4/Q5 projection mapping derived from the measured 1.387× ceiling can
reach 100 tok/s. A result above that boundary would require a different
algorithmic or whole-pipeline dataflow change, not another minor tile variant.

## Decision

**CLOSE THIS BRANCH.** Retain the current B=4 production mapping and all
negative MMQ evidence. The M11-B 100 tok/s gate remains open; this is an
Amdahl ceiling for the tested projection family, not a relaxed success claim.

## Follow-up

Only pursue a new prefill design if it changes the measured work or dependency
structure substantially—for example, a causally valid grouped schedule that
reduces repeated layer-wide work. Do not add another B=4/B=16/B=64 tile
variant without a new end-to-end hypothesis.
