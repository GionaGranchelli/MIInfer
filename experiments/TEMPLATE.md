# EXP-XXXX — <Experiment Title>

**Status:** PROPOSED | RUNNING | KEEP | REJECT | RETEST | INVALID
**Milestone:** M0 | M1 | M2 | M3 | M4 | M5 | M6 | M7
**Author:**
**Date:**
**Baseline commit:**
**Candidate commit:**

---

# 1. Question

State the exact question this experiment is trying to answer.

Keep it narrow.

Example:

> Does splitting a physical Wave64 into two logical 32-lane groups improve Q4 × Q8 GEMV throughput for the target decode shapes on MI50?

Avoid vague questions such as:

> Can we make GEMV faster?

---

# 2. Hypothesis

State the expected outcome before running the experiment.

Example:

> Two logical 32-lane groups will improve throughput because each physical wave can produce two output rows while retaining Wave64 execution, increasing useful work per wave for small-M GEMV.

The hypothesis should be falsifiable.

---

# 3. Motivation

Explain why the experiment is worth running.

Include the measured or suspected bottleneck.

Example:

```text
Target workload:
batch-1 decode

Observed bottleneck:
quantized GEMV

Reason for experiment:
physical Wave64 may be underutilized when all lanes cooperate on one output row
```

Do not justify an experiment only because another project implemented the technique.

---

# 4. Source / Prior Art

List relevant external evidence.

Examples:

```text
iacopPBK/llama.cpp-gfx906
milpster/gfx906-llama-cpp
AMD gfx906 ISA documentation
```

For each source, note what was learned.

Example:

```text
iacopPBK:
uses logical half-wave execution for selected quantized GEMV paths.

Important caveat:
different llama.cpp revision, tensor shapes, and workload.
```

External results are context, not proof.

---

# 5. Target Bottleneck

Classify the expected bottleneck.

Choose one or more:

* HBM bandwidth
* cache behavior
* compute
* instruction throughput
* VGPR pressure
* SGPR pressure
* LDS capacity
* occupancy
* synchronization
* kernel launch overhead
* host/device synchronization
* graph topology
* conversion/packing overhead
* unknown

Current evidence:

```text
<profiling or benchmark evidence>
```

If the bottleneck is unknown, say so explicitly.

---

# 6. Baseline

Describe the implementation being compared against.

## Implementation

```text
<baseline description>
```

## Commit

```text
<git SHA>
```

## Kernel

```text
<kernel/function name>
```

## Relevant configuration

```text
workgroup:
logical wave:
quantization:
accumulator:
tensor layout:
```

The baseline must be runnable independently of the candidate.

---

# 7. Candidate

Describe the proposed change.

## Implementation

```text
<candidate description>
```

## Commit

```text
<git SHA>
```

## Difference from baseline

```text
<minimal technical difference>
```

Avoid bundling unrelated changes.

An experiment should ideally answer one question.

---

# 8. Hardware

## GPU

```text
Model: AMD Instinct MI50 32GB
Architecture: Vega20
ISA: gfx906
Compute Units: 60
Wave size: 64
VRAM: 32 GB HBM2
```

Change these values if the experiment runs on different hardware.

## Runtime state

Record where available:

```text
Power limit:
Observed power:
SCLK:
MCLK/HBM clock:
Temperature before:
Temperature during:
Temperature after:
VRAM usage:
```

If unavailable:

```text
UNAVAILABLE
```

Do not guess.

---

# 9. Host Environment

```text
CPU:
RAM:
Linux distribution:
Kernel:
```

Record anything else that may affect the benchmark.

---

# 10. Software Environment

```text
ROCm:
HIP:
Compiler:
CMake:
Build type:
```

## Compiler flags

```text
<relevant flags>
```

## Environment variables

```text
<relevant environment variables>
```

Do not omit flags that materially affect generated GPU code.

---

# 11. Model / Workload

For model-derived experiments:

```text
Model:
Revision:
Checksum:
Quantization:
Context:
Batch:
Micro-batch:
```

For isolated kernels:

```text
Operation:
Input dtype:
Weight dtype:
Accumulator dtype:
Output dtype:
```

Mark shapes as:

```text
REAL MODEL SHAPE
```

or:

```text
SYNTHETIC
```

---

# 12. Test Matrix

List every benchmark shape before running the experiment.

Example:

| ID |  M |     N |    K | Origin     | Purpose                  |
| -- | -: | ----: | ---: | ---------- | ------------------------ |
| S1 |  1 |  4096 | 4096 | Real model | Decode projection        |
| S2 |  1 | 11008 | 4096 | Real model | FFN projection           |
| S3 |  1 | 31040 | 5120 | Real model | Expert projection        |
| S4 |  4 | 31040 | 5120 | Synthetic  | Small verification batch |

Do not add only shapes where the candidate happens to win after seeing the results.

If the matrix changes during experimentation, document why.

---

# 13. Correctness Method

Describe the trusted reference.

Example:

```text
CPU FP32 implementation
```

or:

```text
rocBLAS FP16 reference
```

or:

```text
known-correct baseline kernel
```

## Validation metrics

Record as appropriate:

```text
max_abs_error:
mean_abs_error:
max_relative_error:
cosine_similarity:
NaN detected:
Inf detected:
```

## Acceptance tolerance

```text
<explicit tolerance>
```

The tolerance must be defined before using the result for performance conclusions.

---

# 14. Correctness Results

| Shape | PASS/FAIL | Max abs error | Mean abs error | Notes |
| ----- | --------- | ------------: | -------------: | ----- |
| S1    |           |               |                |       |
| S2    |           |               |                |       |
| S3    |           |               |                |       |

If correctness fails:

**Experiment status becomes INVALID unless the failure is fixed and the candidate is rerun.**

Do not benchmark known-incorrect code and present the numbers as valid performance evidence.

---

# 15. Benchmark Protocol

## Warm-up

```text
<warm-up iteration count or duration>
```

## Measured runs

```text
<number of measured repetitions>
```

## Iterations per run

```text
<number>
```

## Timing method

```text
HIP events / profiler / other
```

## Ordering

Preferred:

```text
A
B
A
B
A
B
```

For important results also test:

```text
B
A
B
A
B
A
```

Document deviations.

---

# 16. Pre-Run Hardware State

```text
SCLK:
MCLK:
Temperature:
Power:
VRAM:
```

State:

```text
VALID
```

or:

```text
CONTAMINATED
```

Reason if contaminated:

```text
<reason>
```

---

# 17. Raw Results

Do not replace raw measurements with averages.

## Baseline

| Run | Time | Throughput | SCLK | MCLK | Temp | Power |
| --- | ---: | ---------: | ---: | ---: | ---: | ----: |
| A1  |      |            |      |      |      |       |
| A2  |      |            |      |      |      |       |
| A3  |      |            |      |      |      |       |

## Candidate

| Run | Time | Throughput | SCLK | MCLK | Temp | Power |
| --- | ---: | ---------: | ---: | ---: | ---: | ----: |
| B1  |      |            |      |      |      |       |
| B2  |      |            |      |      |      |       |
| B3  |      |            |      |      |      |       |

Machine-readable raw results should be stored separately when available.

---

# 18. Aggregated Results

## Baseline

```text
Mean:
Median:
Min:
Max:
Std dev:
```

## Candidate

```text
Mean:
Median:
Min:
Max:
Std dev:
```

## Delta

For throughput:

```text
((candidate / baseline) - 1) × 100
```

For latency:

```text
(1 - candidate / baseline) × 100
```

Result:

```text
<delta %>
```

---

# 19. Per-Shape Results

| Shape | Baseline | Candidate | Delta | Correctness | Decision |
| ----- | -------: | --------: | ----: | ----------- | -------- |
| S1    |          |           |       |             |          |
| S2    |          |           |       |             |          |
| S3    |          |           |       |             |          |

Do not hide regressions behind an overall average.

---

# 20. Effective Bandwidth

If meaningful for this operation:

```text
Logical bytes read:
Logical bytes written:
Total logical bytes:
Elapsed time:
Effective GB/s:
```

Explain how the byte count was derived.

Do not describe effective bandwidth as literal HBM traffic unless profiling confirms it.

---

# 21. Resource Usage

Record where available:

```text
VGPR:
SGPR:
LDS bytes/workgroup:
workgroup size:
waves/workgroup:
theoretical occupancy:
observed occupancy:
```

Resource metrics help explain results but do not override actual performance.

---

# 22. Generated ISA

For important low-level experiments, inspect generated code.

Record relevant observations:

```text
Expected instruction present:
Unexpected conversion:
Spill detected:
Load pattern changed:
Dot instruction:
DPP instruction:
DS swizzle:
```

Relevant disassembly snippets may be linked or stored separately.

Do not paste huge ISA dumps into this file.

---

# 23. Profiling

Use profiling when required to explain the result.

Potential observations:

```text
HBM utilization:
L2 behavior:
VALU utilization:
wave occupancy:
memory stalls:
instruction stalls:
kernel launch cost:
```

Record only metrics relevant to the hypothesis.

---

# 24. VRAM Impact

```text
Baseline persistent VRAM:
Candidate persistent VRAM:

Baseline peak VRAM:
Candidate peak VRAM:

Delta:
```

If no meaningful difference:

```text
NONE
```

A speed improvement with significant VRAM cost must document the trade-off.

---

# 25. Power / Efficiency Impact

Where measurable:

```text
Baseline average power:
Candidate average power:
```

Optional:

```text
Baseline tokens/J:
Candidate tokens/J:
```

Do not treat raw throughput as the only performance metric when power differs materially.

---

# 26. Unexpected Observations

Record anything that did not match the original hypothesis.

Examples:

* candidate helped only one shape
* compiler did not emit expected instruction
* cache behavior dominated
* register usage unexpectedly increased
* result changed after thermal stabilization
* performance depended on execution order

Unexpected findings are often more valuable than the headline result.

---

# 27. Contaminated Runs

List excluded runs explicitly.

| Run | Reason |
| --- | ------ |
|     |        |

Acceptable reasons include:

* abnormal GPU clocks
* thermal throttling
* competing GPU workload
* benchmark process failure
* known measurement corruption

Do not remove runs solely because they weaken the result.

---

# 28. Interpretation

Explain why the result likely occurred.

Distinguish:

```text
MEASURED FACT
```

from:

```text
HYPOTHESIS
```

Example:

```text
Measured:
Candidate is 7.3% slower across all three representative shapes.

Measured:
VGPR usage increased from 64 to 88.

Hypothesis:
The additional prefetch state increased VGPR pressure without hiding meaningful
HBM latency because the relevant data was already sufficiently cache-resident.
```

Do not present speculative explanation as confirmed fact.

---

# 29. Decision

Choose exactly one:

## KEEP

Use when:

* correctness passes
* result is reproducible
* benefit matters for target workload
* trade-offs are acceptable

## REJECT

Use when:

* correctness passes
* performance regresses or improvement is not worthwhile

## RETEST

Use when:

* result is promising but evidence is insufficient
* variance is too high
* hardware state is questionable
* additional representative shapes are needed

## INVALID

Use when:

* correctness fails
* benchmark methodology is invalid
* runs are materially contaminated

Decision:

```text
KEEP | REJECT | RETEST | INVALID
```

---

# 30. Decision Rationale

State the decision in one concise paragraph.

Example:

```text
REJECT.

The half-wave candidate improved S1 by 2.1% but regressed the two dominant
expert shapes by 6.8% and 8.2%. The aggregate target workload therefore
regresses despite one local win. No further work is justified for the current
kernel configuration.
```

---

# 31. Integration Consequences

If `KEEP`, document:

* files to integrate
* default configuration changes
* required generated tables
* documentation updates
* tests that should become permanent

If `REJECT`, document whether prototype code should be:

* removed
* preserved on experiment branch
* retained behind experimental flag

Avoid leaving rejected optimization paths enabled by default.

---

# 32. Follow-Up Experiments

List only experiments directly motivated by this result.

Example:

```text
EXP-0012:
Retest logical half-wave with reduced accumulator count.

EXP-0013:
Compare packed layout independently from cooperative-width change.
```

Do not turn this section into the entire project backlog.

---

# 33. Files / Artifacts

```text
Raw benchmark results:
Profiler output:
Environment capture:
ISA:
Charts:
Relevant branch:
Relevant commit:
```

Use stable paths where practical.

---

# 34. Final Summary

Complete this section after the experiment ends.

```text
Question:

Result:

Performance delta:

Correctness:

Primary explanation:

Decision:

Next action:
```

This should allow a future contributor to understand the experiment without rereading the entire development history.

---

# Experiment Integrity Rules

1. Define the hypothesis before judging results.
2. Do not change baseline and candidate simultaneously.
3. Keep important benchmark shapes representative.
4. Preserve raw measurements.
5. Validate hardware state.
6. Validate correctness before accepting performance.
7. Do not discard inconvenient measurements without cause.
8. Do not report only the best run.
9. Document regressions.
10. Record negative results.
11. Separate measured facts from explanations.
12. End every experiment with an explicit decision.

The purpose of an experiment is not to prove that an optimization is good.

The purpose is to find out whether it is good.
