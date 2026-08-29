# EXP-0002 — FP16 GEMV Baseline on Qwen3-8B Shapes

**Status:** PROPOSED
**Milestone:** M1
**Author:**
**Date:** 2026-08-29
**Baseline commit:** PENDING — MI50 execution and clean benchmark commit required
**Candidate commit:** Not applicable — baseline measurement only

---

# 1. Question

What is the correctness and timing baseline for an FP16 GEMV implementation on
the exact dense Qwen3-8B projection shapes used by the first MIInfer control
model?

# 2. Hypothesis

The six projection families will expose materially different GEMV behavior as
their output dimension changes, while the square Q/O shapes and tall FFN
shapes provide representative baseline cases for later gfx906-specific work.

# 3. Motivation

M0 freezes the external control model and the benchmark laboratory. M1 needs
actual model dimensions rather than synthetic dimensions before any kernel
optimization is proposed.

# 4. Source / Prior Art

* `docs/qwen3-8b-baseline.md` — pinned official config and derived shapes.
* `docs/reference-baseline.md` — pinned external gfx906 reference.
* `docs/benchmarking.md` — correctness, repetition, and contamination rules.

# 5. Target Bottleneck

Unknown until the baseline is measured. Candidate categories are HBM
bandwidth, cache behavior, compute throughput, instruction throughput, VGPR
pressure, and occupancy.

# 6. Baseline

## Implementation

The first baseline is a correctness-trusted, unoptimized FP16 GEMV path to be
implemented and measured in the next task. This scaffold does not implement
that path.

## Commit

```text
PENDING
```

## Relevant configuration

```text
input vector: length K, FP16
weights: FP16, M x K
accumulator: FP32 reference and explicitly recorded candidate choice
batch: 1
architecture: gfx906
```

# 7. Candidate

None in this scaffold. Optimization candidates must be separate, correctness-
verified experiments with an A/B baseline.

# 8. Hardware

```text
GPU: AMD Instinct MI50 32GB
Architecture: gfx906 / Vega20
Wave size: 64
Execution: single GPU
```

Runtime state is required from active telemetry and environment artifacts for
every measured run.

# 9. Host / Software Environment

Record CPU, RAM, Linux kernel, ROCm, HIP compiler, compiler flags, relevant
environment variables, and exact MIInfer commit at execution time.

# 10. Model / Workload

```text
Model: Qwen/Qwen3-8B
Revision: b968826d9c46dd6066d109eabc6255188de91218
Source dtype: bfloat16
Benchmark dtype: FP16 representation, pending reproducible GGUF artifact
Batch: 1
```

# 11. Test Matrix — REAL MODEL SHAPES

The matrix uses `M x K`, where the input vector has length `K` and the output
has length `M`.

| ID | Projection | M | K | Origin |
| --- | --- | ---: | ---: | --- |
| Q | Q projection | 4096 | 4096 | REAL MODEL SHAPE |
| K | K projection | 1024 | 4096 | REAL MODEL SHAPE |
| V | V projection | 1024 | 4096 | REAL MODEL SHAPE |
| O | Output projection | 4096 | 4096 | REAL MODEL SHAPE |
| G | FFN gate projection | 12288 | 4096 | REAL MODEL SHAPE |
| U | FFN up projection | 12288 | 4096 | REAL MODEL SHAPE |
| D | FFN down projection | 4096 | 12288 | REAL MODEL SHAPE |

# 12. Correctness

Compare GPU results against a deterministic host FP32 accumulation reference
using the same FP16 inputs and weights. Record maximum absolute error, maximum
relative error, and PASS/FAIL for every shape before accepting timings.

# 13. Benchmark Protocol

Use HIP events around the kernel, discard documented warm-up iterations, retain
all measured samples, and report mean, median, minimum, maximum, and standard
deviation. Repeat each shape under the same MI50 power/clock conditions.

# 14. Results

Not run. No GEMV implementation or timing is part of this task.

# 15. Contamination / Interpretation

Inspect active telemetry for low or unstable SCLK/MCLK, thermal throttling,
power-limit changes, and competing workloads. Retain contaminated artifacts and
exclude them explicitly from conclusions.

# 16. Decision

`PROPOSED` until the baseline is implemented, correctness-verified, and run on
the physical MI50.

# 17. Follow-up

Implement only the smallest correctness-first FP16 GEMV baseline, then measure
the seven shapes before proposing gfx906-specific optimization.
