# EXP-0001 — Benchmark Harness Validation

**Status:** PROPOSED
**Milestone:** M0
**Author:**
**Date:** 2026-08-29
**Baseline commit:** UNAVAILABLE — repository metadata is not present in this workspace
**Candidate commit:** UNAVAILABLE — repository metadata is not present in this workspace

---

# 1. Question

Can the MIInfer benchmark infrastructure produce correct, repeatable HIP
timings on gfx906 while recording enough environment state to detect invalid
benchmark conditions?

# 2. Hypothesis

The project-owned vector-add benchmark will produce correct results and stable
HIP-event timing samples when run repeatedly on a correctly configured MI50,
and the before/after environment records will expose relevant hardware-state
fields or mark them unavailable.

# 3. Motivation

This is an infrastructure validation experiment. No LLM kernel or inference
runtime is involved. The initial operation is deliberately synthetic and is
not evidence about MIInfer inference performance.

# 4. Source / Prior Art

The experiment follows the repository benchmark and hardware guidance:

* `docs/benchmarking.md` — warm-up, repeated samples, HIP timing, raw output,
  and contamination rules.
* `docs/hardware.md` — gfx906 validation and clock/temperature/power capture.

# 5. Target Bottleneck

Kernel launch overhead; the benchmark is not intended to identify an
inference bottleneck.

# 6. Baseline

## Implementation

`miinfer-bench` project-owned vector-add HIP kernel.

## Kernel

`miinfer::launch_vector_add`

## Relevant configuration

```text
operation: vector add
architecture: gfx906
workgroup: 256 threads
timing: HIP events
```

# 7. Candidate

None. This experiment validates the initial harness.

# 8. Hardware

```text
GPU: AMD Instinct MI50 32GB
Architecture: gfx906 / Vega20
Wave size: 64
Runtime state: record from environment-before.json and environment-after.json
```

# 9. Host Environment

Record from the captured JSON. Values not exposed by the system remain
`UNAVAILABLE`.

# 10. Software Environment

Record ROCm, HIP compiler, CMake, build type, compiler flags, and relevant
environment variables from the run.

# 11. Model / Workload

```text
Operation: vector add
Input dtype: FP32
Output dtype: FP32
Shape: synthetic, 1,048,576 elements by default
```

# 12. Test Matrix

The initial run matrix is one synthetic smoke shape. Future executions should
add sizes only when the reason is documented.

| ID | Elements | Origin    | Purpose            |
| -- | -------: | --------- | ------------------ |
| S1 | 1048576  | Synthetic | Harness validation |

# 13. Correctness Method

Compare every output element with the deterministic FP32 host expectation
`1.25 + 2.5 = 3.75`, reject NaN/Inf, and use an absolute tolerance of `1e-6`.

# 14. Correctness Results

To be filled by execution on a usable MI50.

# 15. Benchmark Protocol

```text
Warm-up: 5 launches by default
Measured runs: 100 launches by default
Timing: HIP events around the asynchronous kernel launch
Ordering: single benchmark; no A/B candidate exists
```

# 16. Pre-Run Hardware State

Read from `environment-before.json`; do not guess unavailable fields.

# 17. Raw Results

The benchmark JSON retains aggregate values. Raw per-iteration samples are a
follow-up improvement if this experiment requires variance analysis.

# 18. Aggregated Results

To be filled by execution. The result JSON reports mean, median, min, max, and
standard deviation in microseconds.

# 19. Per-Shape Results

S1: pending execution.

# 20. Effective Bandwidth

Not used for this launch/timing validation. The operation is not a bandwidth
claim.

# 21. Resource Usage

Not collected by the initial harness.

# 22. Generated ISA

Not collected by the initial harness.

# 23. Profiling

Not collected by the initial harness.

# 24. VRAM Impact

Record the environment fields if available; no persistent allocation is part
of the experiment.

# 25. Power / Efficiency Impact

Record available before/after state. No energy conclusion is drawn.

# 26. Unexpected Observations

To be filled after the physical MI50 run.

# 27. Contaminated Runs

To be filled after execution. Retain the reason for any invalid run.

# 28. Interpretation

Pending. Separate measured facts from explanations after execution.

# 29. Decision

`RETEST` or `INVALID` until executed on the required hardware.

# 30. Decision Rationale

No benchmark result is fabricated in this scaffold.

# 31. Integration Consequences

None. This experiment validates infrastructure only.

# 32. Follow-Up Experiments

After this harness is executed, begin the smallest representative FP16 GEMV
baseline experiment if the project still has a usable MI50 environment.

# 33. Files / Artifacts

```text
bench/results/<run-id>/environment-before.json
bench/results/<run-id>/result.json
bench/results/<run-id>/environment-after.json
```

# 34. Final Summary

```text
Question: pending physical MI50 execution
Result: pending
Correctness: pending
Decision: RETEST or INVALID
```
