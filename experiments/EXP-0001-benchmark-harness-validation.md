# EXP-0001 — Benchmark Harness Validation

**Status:** RETEST
**Milestone:** M0
**Author:**
**Date:** 2026-08-29
**Baseline commit:** 990b1912d72b — bootstrap implementation at task start
**Candidate commit:** `a344ff626b12` — physical pilot attempted; commit was dirty due pre-existing untracked `.claude/` and `CLAUDE.md`

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
Active telemetry: rocm-smi JSONL sampling every 250 ms by default, with millisecond UTC timestamps
```

# 16. Pre-Run Hardware State

Read from `environment-before.json`, `telemetry.jsonl`, and
`environment-after.json`; do not guess unavailable fields. A run with
unexpectedly low clocks, thermal throttling, or another GPU workload must be
marked contaminated.

# 17. Raw Results

The benchmark JSON retains every per-iteration HIP-event sample in
`samples_us`, along with aggregate values. The runner additionally stores
active-run GPU telemetry as JSON Lines.

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
bench/results/<run-id>/telemetry.jsonl
bench/results/<run-id>/environment-after.json
```

# 34. Final Summary

```text
Question: pending physical MI50 execution
Result: pending
Correctness: pending
Decision: RETEST or INVALID
```

# 35. Local Execution Gate

An execution attempt on 2026-08-29 completed the repository-side checks but
could not reach HIP execution. `/dev/kfd` was present and the user had the
`render` and `video` groups, but `rocminfo` failed during HSA initialization
with `HSA_STATUS_ERROR: A generic error has occurred`; HIP then reported no
ROCm-capable device. `rocm-smi` exposed two AMD devices: card0 `gfx802` with
8589934592 bytes of VRAM and card1 `gfx906` with 34342961152 bytes of VRAM,
connected by PCIe. The CTest host-only test passed; the GPU-required test and
benchmark failed explicitly before producing timing values.

Preflight details: `/dev/kfd` was `crw-rw-rw-. root render`, render nodes
`/dev/dri/renderD128` and `renderD129` were present, and the executing user was
in both `render` and `video`. The exact `rocminfo` failure was emitted from
`rocminfo.cc:1324` and returned the generic HSA error above. This is recorded
as a platform/runtime blocker, not a reason to broaden device validation.

No timing result is recorded. The experiment remains `PROPOSED` and must be
executed on a usable physical MI50 before it receives a benchmark decision.

The Task 2 repository changes are not represented by a clean commit yet, so
there is no eligible provenance commit for physical benchmark evidence. When
the MI50 run is performed, record the exact clean MIInfer commit, dirty state,
release preset, benchmark command, and artifact directories here.

## 36. Task 3 Platform-Recovery Pilot

The requested pilot was attempted on 2026-08-29 with:

```text
run: bench/results/20260829T122124Z-874753
command: scripts/run-bench.sh ./build/mi50-release/miinfer-bench --warmup 5 --iterations 100 --elements 1048576
exit: 1
failure: hipGetDeviceCount failed: no ROCm-capable device is detected
```

The runner retained `environment-before.json`, `telemetry.jsonl`, and
`environment-after.json`. Telemetry captured one active sample while the
benchmark was starting:

```text
gfx802: 398 MHz SCLK, 1250 MHz MCLK, 57 C, 30.551 W average package power
gfx906: 930 MHz SCLK, 350 MHz MCLK, 35--36 C, 21 W socket package power
```

No `result.json` or HIP timing samples were produced. This is a retained
failed pilot, not one of the five valid EXP-0001 runs. Five independent runs
must wait until ROCr/HIP recovery and a clean MIInfer commit are available.

The platform diagnosis and recovery constraints are recorded in
[`docs/platform-mi50.md`](../docs/platform-mi50.md).
