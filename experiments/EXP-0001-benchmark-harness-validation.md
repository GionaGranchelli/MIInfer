# EXP-0001 — Benchmark Harness Validation

**Status:** KEEP
**Milestone:** M0
**Author:**
**Date:** 2026-08-29
**Baseline commit:** 990b1912d72b — bootstrap implementation at task start
**Candidate commit:** `6ae036d0d29d` — clean MI50 execution commit

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

All five accepted runs passed deterministic FP32 correctness checks with no
NaN/Inf output.

# 15. Benchmark Protocol

```text
Warm-up: 5 launches
Measured runs: 100000 launches
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

Five independent Release runs completed on the clean commit
`6ae036d0d29d8a09f058f3a10adc0219badcb70b`. Every run retained 100000 raw
samples, correctness PASS, and nine telemetry records. The median was
23.52 us in every run; the cross-run mean of the five per-run means was
23.7875 us.

# 19. Per-Shape Results

S1 (`1048576` elements):

| Run | Mean (us) | Median (us) | Stddev (us) | Min (us) | Max (us) | Telemetry |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `20260829T192840Z-21891` | 23.750313 | 23.52 | 2.560998 | 22.879001 | 260.320008 | 9 |
| `20260829T192845Z-22171` | 23.761788 | 23.52 | 2.484629 | 22.879999 | 105.120003 | 9 |
| `20260829T192850Z-22451` | 23.863990 | 23.52 | 2.633991 | 23.039000 | 101.599999 | 9 |
| `20260829T192856Z-22728` | 23.795254 | 23.52 | 2.557975 | 22.879999 | 164.799005 | 9 |
| `20260829T192901Z-23009` | 23.766348 | 23.52 | 2.458129 | 22.879999 | 76.480001 | 9 |

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

The MI50 reached approximately 1725 MHz SCLK and 1000 MHz MCLK during active
sampling, versus approximately 925--930 MHz and 350 MHz at idle. Active
temperature was approximately 35--43 C and package/socket power was about
125--129 W. The raw samples retain occasional timing outliers; they were not
deleted or silently excluded.

# 27. Contaminated Runs

None of the five accepted runs was marked contaminated. The earlier 100-
iteration pilot is retained separately because it produced only one useful
telemetry sample and is not part of the accepted set.

# 28. Interpretation

The harness produced correct, repeatable HIP-event timings and usable active
telemetry on the isolated physical MI50. This validates the benchmark
infrastructure only; it makes no inference-performance claim.

# 29. Decision

`KEEP`

# 30. Decision Rationale

Five independent clean Release runs passed correctness, retained raw samples,
and captured active telemetry. The accepted evidence is reproducible from the
clean MIInfer commit and the listed run directories.

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
Question: Can the harness produce valid repeatable gfx906 measurements?
Result: PASS on the physical MI50 with gfx802 isolated
Correctness: PASS in all five accepted runs
Decision: KEEP
```

# 35. Historical failed execution gate — superseded by Section 37

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

No timing result was recorded for this historical attempt. This failed gate is
retained for provenance and is superseded by the accepted physical MI50
execution documented in Section 37; it does not change the experiment's
current `KEEP` decision.

At the time of this historical attempt, the Task 2 repository changes were not
represented by a clean commit, so it was not eligible provenance for physical
benchmark evidence. The later accepted execution records the clean commit,
release preset, command, and artifact directories.

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
failed pilot, not one of the five valid EXP-0001 runs. It was superseded by the
accepted clean-commit execution documented in Section 37.

The platform diagnosis and recovery constraints are recorded in
[`docs/platform-mi50.md`](../docs/platform-mi50.md).

## 37. Accepted MI50 execution

The five runs listed in Section 19 are the accepted EXP-0001 result. They used
the MI50-only KFD configuration from the confirmed gfx802 isolation test and
the clean MIInfer commit shown above. The result is `KEEP`; it validates the
harness and laboratory state, not LLM inference performance.
