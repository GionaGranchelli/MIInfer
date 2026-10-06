# M31-0004 — Stability and Release Gate

## Scope

Repeat the frozen M31-0001 through M31-0003 workloads without changing source,
model, image, GPU policy, or benchmark inputs. This stage produces the explicit
single-MI50 qualification decision.

## Required repetitions

```text
10 cold runs
10 warm/reuse runs
10 canonical agent workload runs
```

## Required statistics and failure accounting

```text
median, p95, min, max, coefficient_of_variation
VRAM drift
correctness failures
GPU errors
OOM
state leaks
```

The gate emits exactly one of:

```text
SINGLE_MI50_QUALIFIED
SINGLE_MI50_NOT_QUALIFIED
```

Qualification records weaknesses rather than reopening M30 optimization. The
result becomes the N=1 invariant for V3: dual-MI50 work must preserve N=1
correctness and must not materially regress the frozen single-MI50 path.

## Evidence

```text
M31_BASELINE_SHA=99840d10eabe5e64fe2564d34d3d3fcd42b43521
M31_0004_GATE_SHA=7ee03f8678fd59c82c88246e5b5250464529c216
M31_0004_RESULT=SINGLE_MI50_NOT_QUALIFIED
```

The aggregation gate is implemented in
`bench/m31_0004_stability_release_gate.py`. It requires ten recorded runs in
each category and fails closed when repetitions, correctness evidence, or
upstream M31 qualification are missing. The current decision is
`SINGLE_MI50_NOT_QUALIFIED` because M31-0001 has a 64K TG128 parity failure,
M31-0002 has a prefix-reuse parity failure, and the native M31-0003 workload
has an output-parity failure. The native workload does exercise COW, so the
earlier HTTP-interface limitation is no longer the blocker. No release claim
is made from the incomplete repetition set.
