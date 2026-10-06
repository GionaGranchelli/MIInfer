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
M31_0004_GATE_SHA=4b4e61a8bd178d8378dcb3f3f18f7b56069ab89e
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

The final stability campaign completed all ten native workload repetitions.
All ten deterministically failed output parity, with no process-reported GPU
error or OOM. Across the repetitions, cold wall time was median 256730.368 ms
(p95 256764.668 ms), persistent wall time was median 47165.764 ms (p95
47232.705 ms), and combined agent wall time was median 303898.895 ms (p95
303978.661 ms). Peak VRAM was constant at 26971482112 bytes (zero observed
drift). The gate artifact is
`results/m31-0004-release-gate-final.json` (SHA-256
`09695a5adae63666f9cb6444f459ce5f3956b0399c929edf004d8829dc158ea4`); the
run log is `results/m31-0004-native-runs/stability.log` (SHA-256
`c24121b153c6a9d9fb58507cf3a09c18f73c82c51fde940edd1f697a4df73ad0`).
