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
M31_0004_GATE_SHA=<filled after implementation>
M31_0004_RESULT=<SINGLE_MI50_QUALIFIED or SINGLE_MI50_NOT_QUALIFIED>
```
