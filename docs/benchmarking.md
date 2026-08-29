# MIInfer Benchmarking Standard

This document defines how MIInfer performance measurements must be collected, compared, interpreted, and recorded.

Performance is a primary project goal, so benchmarking is part of the engineering contract rather than an optional validation step.

A performance claim is valid only when the compared runs are sufficiently controlled, reproducible, and correctness-verified.

---

# 1. Benchmarking Objective

MIInfer exists to determine whether a runtime specialized for gfx906 can materially outperform the strongest practical existing inference implementations on the same hardware.

The benchmark process must therefore distinguish real software improvements from:

* GPU clock variation
* thermal behavior
* power-limit changes
* cache state
* ROCm differences
* compiler differences
* model differences
* quantization differences
* context differences
* measurement noise
* workload differences

The default assumption is:

> A performance difference is unproven until controlled measurement supports it.

---

# 2. Benchmark Targets

MIInfer benchmarks operate at several levels.

## 2.1 Primitive benchmarks

Examples:

* DPP reductions
* swizzle primitives
* packed integer dot products
* memory load patterns
* quantization primitives

Primary metrics:

* latency
* throughput
* instruction behavior
* VGPR use
* LDS use
* occupancy where meaningful

---

## 2.2 Kernel benchmarks

Examples:

* GEMV
* skinny GEMM
* RMSNorm
* RoPE
* activation quantization
* attention
* MoE projections

Primary metrics:

* kernel time
* effective bandwidth
* throughput
* numerical error
* relevant profiler metrics

---

## 2.3 Model-component benchmarks

Examples:

* QKV projection stage
* one transformer block
* one MoE block
* attention at a fixed context
* complete decode layer stack

These help determine whether microkernel wins survive composition.

---

## 2.4 End-to-end benchmarks

Primary categories:

### Prompt processing / prefill

Measure prompt-ingestion throughput independently.

Typical metric:

```text
tokens / second
```

### Token generation / decode

Measure autoregressive generation separately.

Typical metric:

```text
tokens / second
```

Never combine prefill and generation into one throughput number.

---

# 3. Correctness Is a Benchmark Prerequisite

A faster implementation that fails correctness is not a valid benchmark candidate.

Before comparing performance, validate the candidate at the appropriate level.

Examples:

### Kernel

Compare against trusted reference output.

Record where useful:

* max absolute error
* mean absolute error
* max relative error
* cosine similarity

### Model component

Compare:

* intermediate tensor values
* logits
* top token

### End-to-end inference

Validate:

* deterministic generation
* short sequence behavior
* long generation
* long-context stability

Performance results from a correctness-failing candidate must be clearly marked invalid.

---

# 4. Hardware Baseline

Initial benchmark hardware:

```text
AMD Instinct MI50 32GB
gfx906
Vega20
single GPU
```

Benchmark results must not silently mix hardware.

If results come from Radeon VII, MI60, another MI50 variant, or a different GPU, record that explicitly.

---

# 5. Required Environment Metadata

Important benchmark runs should record at minimum:

## Hardware

* GPU model
* gfx architecture
* VRAM
* GPU clock
* HBM/memory clock where available
* power
* configured power limit
* temperature
* GPU utilization
* VRAM utilization

## Host

* CPU
* RAM
* Linux distribution
* kernel version

## Software

* MIInfer commit
* compiler version
* HIP compiler version
* ROCm version
* relevant runtime libraries
* build type
* relevant compiler flags
* relevant environment variables

## Workload

* model
* model checksum or revision
* quantization
* tensor representation
* prompt/input
* context length
* batch size
* micro-batch size where relevant
* generation length
* sampling configuration

If the benchmark cannot reproduce these conditions later, its evidentiary value is reduced.

---

# 6. GPU Clock Validation

gfx906 benchmark results are sensitive to hardware state.

Before trusting a benchmark, verify that expected clocks are reached during active execution.

Do not trust idle clock readings alone.

Record, where available:

```text
sclk
mclk
temperature
power
```

during workload execution.

A run should be treated as contaminated if:

* GPU clocks remain unexpectedly low
* thermal throttling occurs
* power throttling differs materially between A and B
* the GPU enters an abnormal performance state
* another process materially consumes the GPU

Contaminated runs should be documented, not quietly mixed into the result set.

---

# 7. Warm-Up

Never use the first execution as the primary timing result unless measuring cold-start behavior intentionally.

Warm-up may affect:

* code loading
* cache state
* GPU clocks
* runtime initialization
* graph setup
* allocator behavior

Each benchmark should define an explicit warm-up policy.

Example:

```text
5 warm-up iterations
20 measured iterations
```

The exact number depends on workload duration and stability.

---

# 8. Repetition

Single-run measurements are insufficient for performance claims.

Important benchmarks should include repeated runs.

At minimum collect enough samples to understand normal variance.

Example:

```text
A1
A2
A3
A4
A5
```

For very stable long-running workloads, fewer runs may be acceptable.

For microbenchmarks, significantly more iterations may be required.

---

# 9. Interleaved A/B Testing

For comparing small performance differences, prefer interleaved execution:

```text
A
B
A
B
A
B
```

over:

```text
A
A
A
B
B
B
```

Interleaving reduces bias caused by:

* temperature drift
* clock drift
* host load changes
* persistent runtime state
* environmental changes over time

For important comparisons, A and B should be built and available before the test begins.

Avoid rebuilding between every measured run unless the benchmark explicitly studies build differences.

---

# 10. Reversed Ordering

For particularly important results, repeat the comparison with reversed ordering:

```text
B
A
B
A
B
A
```

If a claimed improvement disappears when order changes, investigate before accepting the result.

---

# 11. Statistics

Raw results must be retained.

Useful aggregate statistics include:

* arithmetic mean
* median
* minimum
* maximum
* standard deviation where meaningful

The median is often useful for noisy latency measurements.

The mean remains useful for total throughput comparisons.

Do not report only the best run.

---

# 12. Performance Delta

For throughput metrics where higher is better:

```text
delta % = ((candidate / baseline) - 1) × 100
```

Example:

```text
baseline = 100 tok/s
candidate = 115 tok/s

delta = +15%
```

For latency where lower is better:

```text
delta % = (1 - candidate / baseline) × 100
```

Example:

```text
baseline = 100 us
candidate = 80 us

improvement = +20%
```

Always make metric direction clear.

---

# 13. Benchmark Stability

A benchmark result should not be promoted to a project conclusion if normal variance is of similar magnitude to the claimed improvement.

Example:

```text
observed gain: +1.8%
normal run variance: ±2.5%
```

Result:

```text
INCONCLUSIVE
```

Do not round noisy differences into confident claims.

---

# 14. Baseline Selection

MIInfer should compare against the strongest reproducible relevant implementation available.

Do not intentionally select a weak baseline.

Primary external baseline should normally be:

* a well-tuned gfx906 llama.cpp implementation
* running equivalent model/quantization/workload
* using a validated gfx906 software stack

Stock llama.cpp may also be recorded for context, but it should not be the sole baseline if materially better gfx906 implementations exist.

---

# 15. Baseline Pinning

Every benchmark series must pin the baseline.

Record:

```text
repository
commit SHA
build flags
runtime arguments
ROCm version
environment variables
```

A moving `master` branch is not a reproducible baseline.

---

# 16. Model Equivalence

End-to-end comparisons should use equivalent models whenever possible.

Record:

* model name
* exact revision
* checksum
* quantization
* model size
* tensor format

If MIInfer uses a custom packed representation, document how it was produced from the baseline model.

Packing itself may be an optimization, but the underlying numerical model should remain equivalent.

---

# 17. Quantization Equivalence

Do not compare:

```text
MIInfer Q4
```

against:

```text
baseline Q8
```

and claim a pure runtime speed advantage.

If quantization differs, clearly separate:

* runtime gain
* quantization gain
* memory gain
* quality difference

Prefer identical numerical formats for architecture comparisons.

---

# 18. Power Equivalence

Performance comparisons must consider power limits.

A candidate running at 300 W should not be directly presented as superior to a baseline constrained to 225 W without disclosure.

Record:

* configured power cap
* observed power

Where useful, report:

```text
tokens / joule
```

in addition to raw throughput.

---

# 19. Temperature Equivalence

Avoid comparing:

```text
baseline at 85°C
candidate at 45°C
```

without accounting for thermal effects.

Allow hardware to reach stable operating temperature for sustained workloads.

Where necessary, perform alternating runs long enough to keep thermal state comparable.

---

# 20. Context Length

Context length is a first-class benchmark dimension.

The dominant workload may change with context.

At minimum, model benchmarks should eventually include representative regimes such as:

```text
short
~512 tokens

medium
~8K tokens

long
~32K tokens
```

Exact values may change based on model and VRAM capacity.

Do not extrapolate short-context decode performance to long context without measurement.

---

# 21. Prompt Processing Benchmarks

Initial standard prompt-processing cases should include, where practical:

```text
PP512
PP2048
PP8192
```

Record:

* prompt size
* batch
* micro-batch
* context before prompt
* elapsed time
* prompt tokens/s

Prompt-processing results should not include unrelated model-loading time.

---

# 22. Decode Benchmarks

Initial token-generation cases should include, where practical:

```text
TG128
TG512
TG1024
```

Record:

* initial context
* number of generated tokens
* total generation time
* tokens/s

The benchmark should distinguish:

```text
fresh / short context
```

from:

```text
deep-context generation
```

---

# 23. Time to First Token

Where end-user latency matters, record TTFT separately.

TTFT may include:

* prompt processing
* decode setup
* graph setup where applicable
* first-token selection

The exact definition must be documented for each benchmark suite.

---

# 24. Kernel Timing

GPU kernel timing should use an appropriate GPU timing mechanism.

Avoid host wall-clock timing for very short kernels unless launch overhead is intentionally part of the measurement.

Where possible, distinguish:

```text
kernel execution time
```

from:

```text
host launch + runtime overhead
```

Both may matter, but they answer different questions.

---

# 25. Effective Memory Bandwidth

For bandwidth-oriented kernels, estimate effective bandwidth when the byte count can be meaningfully defined.

Conceptually:

```text
effective GB/s =
bytes logically moved / elapsed time
```

Document how the byte count was derived.

Avoid presenting effective bandwidth as literal physical HBM traffic unless profiler evidence supports that interpretation.

---

# 26. Profiler Use

Profiling should follow measurement, not replace it.

Use profiling to explain results.

Potential metrics include:

* kernel duration
* VGPR count
* SGPR count
* LDS use
* occupancy
* cache behavior
* memory transactions
* instruction count
* wave occupancy
* stalls

Do not optimize a profiler metric merely because it looks undesirable.

Examples:

```text
lower VGPR count
```

is not automatically better.

```text
higher occupancy
```

is not automatically better.

The actual throughput result remains authoritative.

---

# 27. Compilation Effects

Compiler changes can materially affect gfx906 code generation.

Record:

* compiler version
* optimization level
* architecture flags
* fast-math options
* LTO state
* relevant HIP flags

If a performance change results mainly from compiler configuration rather than source changes, record it as such.

---

# 28. Cache Effects

Some microbenchmarks can accidentally measure cache-resident behavior that does not reflect real model execution.

When benchmarking memory-heavy kernels, consider:

* working-set size
* L2 capacity
* repeated tensor reuse
* cache flush or rotation strategy where appropriate
* whether real inference reuses the data similarly

Do not optimize specifically for an unrealistic benchmark cache state.

---

# 29. Synthetic vs Representative Shapes

Synthetic benchmarks are acceptable for architectural exploration.

However, project decisions should prioritize tensor shapes from actual supported models.

Every major kernel benchmark should identify whether the shape is:

```text
REAL MODEL SHAPE
```

or:

```text
SYNTHETIC
```

Synthetic wins do not automatically justify runtime complexity.

---

# 30. Microbenchmark to Model Validation

Important kernel wins should progress through:

```text
microbenchmark
      ↓
representative tensor shape
      ↓
model component
      ↓
end-to-end inference
```

A kernel-level improvement may disappear because of:

* launch overhead
* synchronization
* data conversion
* cache interactions
* unrelated bottlenecks
* model execution topology

Do not assume gains compose linearly.

---

# 31. Experiment IDs

Important performance investigations should use an experiment identifier:

```text
EXP-0001
EXP-0002
...
```

Benchmark output should include the experiment ID where practical.

Example directory:

```text
bench/results/EXP-0004/
├── environment.json
├── baseline.csv
├── candidate.csv
├── summary.json
└── notes.md
```

Raw result files should normally remain reproducible artifacts.

Whether large benchmark outputs are committed to Git or stored externally can be decided later.

---

# 32. Suggested Raw Result Schema

Machine-readable benchmark output should contain fields similar to:

```json
{
  "experiment": "EXP-0004",
  "commit": "<sha>",
  "gpu": "AMD Instinct MI50",
  "gfx": "gfx906",
  "rocm": "<version>",
  "benchmark": "q4_q8_gemv",
  "m": 1,
  "n": 31040,
  "k": 5120,
  "iterations": 100,
  "median_us": 0,
  "mean_us": 0,
  "min_us": 0,
  "max_us": 0,
  "max_abs_error": 0
}
```

The exact schema may evolve.

Keep it simple enough to inspect manually.

---

# 33. Environment Capture

Benchmark automation should eventually capture environment information before each run.

Preferred workflow:

```text
benchmark starts
      ↓
capture environment
      ↓
validate GPU state
      ↓
warm-up
      ↓
measure
      ↓
capture post-run state
      ↓
write raw output
```

This makes abnormal clock or temperature behavior visible.

---

# 34. Run Contamination

Possible contamination reasons include:

* GPU clock stuck below expected state
* thermal throttling
* competing GPU process
* host under exceptional load
* VRAM pressure from another process
* failed previous process leaving abnormal runtime state
* system suspend/resume behavior
* unexpected power-cap change

Do not silently delete contaminated measurements.

Record:

```text
INVALID / CONTAMINATED
```

and the reason.

---

# 35. Outlier Handling

Do not remove an outlier solely because it weakens the result.

Outlier removal requires a documented reason such as:

* system interruption
* known hardware-state failure
* benchmark process error
* obvious measurement corruption

Otherwise keep the data.

Use robust statistics where appropriate.

---

# 36. Performance Regression Policy

A change intended for correctness or architecture may still introduce a performance regression.

If so:

1. measure it
2. quantify it
3. determine why
4. decide whether the trade-off is acceptable

Do not assume performance can always be recovered later.

For performance-sensitive paths, meaningful regressions should normally block the change unless justified.

---

# 37. VRAM Measurement

End-to-end benchmarks should record memory usage.

Track where possible:

```text
weights
KV
persistent runtime
temporary workspace
optional caches
peak VRAM
```

An optimization that increases speed but materially reduces context capacity must document that trade-off.

---

# 38. Context Capacity

Where relevant, benchmark maximum practical context separately from throughput.

Record:

* model
* quantization
* KV format
* runtime overhead
* maximum context
* remaining VRAM margin

Avoid operating so close to capacity that minor allocator variation causes instability unless explicitly testing the limit.

---

# 39. Energy Efficiency

Where reliable power measurement is available:

```text
tokens/joule =
tokens generated / energy consumed
```

Approximate versions may use:

```text
tokens/s / watts
```

but must be labeled as approximate.

Raw speed and efficiency are both useful.

---

# 40. Acceptance Categories

Every performance experiment should end in one of four states.

## KEEP

The candidate is correct and provides a meaningful benefit for the target workload.

## REJECT

The candidate is correct but provides no worthwhile benefit or regresses performance.

## RETEST

The result is promising but evidence is insufficient or contaminated.

## INVALID

Correctness failed or benchmark conditions were invalid.

---

# 41. Performance Claim Language

Use precise language.

Prefer:

```text
+12.4% median TG over baseline across 6 interleaved runs
```

over:

```text
much faster
```

Prefer:

```text
no statistically useful difference was observed
```

over:

```text
same speed
```

when measurements are noisy.

Avoid claiming architectural superiority from one kernel result.

---

# 42. Benchmark Result Template

A benchmark conclusion should be understandable in isolation.

Recommended format:

```text
Experiment:
EXP-XXXX

Baseline:
<implementation / commit>

Candidate:
<implementation / commit>

Hardware:
MI50 32GB / gfx906

ROCm:
<version>

Workload:
<exact model or kernel shape>

Correctness:
PASS

Baseline:
<aggregated result>

Candidate:
<aggregated result>

Delta:
<percentage>

VRAM delta:
<if applicable>

Power delta:
<if applicable>

Decision:
KEEP / REJECT / RETEST

Interpretation:
<why the result probably occurred>
```

---

# 43. Initial MIInfer Benchmark Matrix

During M1/M2, the exact matrix will be refined from the selected target model.

Initial categories should include:

## GEMV

* FP16
* initial quantized baseline
* candidate Q4 × Q8
* other target quant formats

## Cooperative execution

* Wave64
* logical 32-lane grouping
* alternative grouping only when justified

## Memory layout

* canonical
* packed candidate

## Normalization

* CPU/reference
* straightforward HIP
* optimized gfx906

## Activation quantization

* standalone cost
* quantize-once/reuse scenario

Attention and MoE benchmarks should be added once target-model profiling provides representative shapes.

---

# 44. End-to-End Reference Matrix

Once M4/M5 is reached, maintain a stable comparison matrix.

Example:

| Test             | Baseline | MIInfer |
| ---------------- | -------: | ------: |
| PP512            |          |         |
| PP2048           |          |         |
| PP8192           |          |         |
| TG128 short ctx  |          |         |
| TG512 short ctx  |          |         |
| TG1024 short ctx |          |         |
| TG128 @ 8K       |          |         |
| TG128 @ 32K      |          |         |
| TTFT             |          |         |
| Peak VRAM        |          |         |
| Power            |          |         |
| tokens/J         |          |         |

Do not fill unsupported cases with estimates.

---

# 45. Benchmark Changes

Benchmark methodology itself must be version-controlled.

If a benchmark changes materially:

* document what changed
* do not mix old and new results as directly equivalent
* rerun important baselines where necessary

Examples of material changes:

* timer implementation
* warm-up policy
* model revision
* prompt
* context definition
* compiler flags
* KV format

---

# 46. Reproducibility

A contributor should eventually be able to reproduce an important benchmark using:

```text
commit
environment definition
build command
benchmark command
model revision
experiment record
```

Perfect cross-machine reproducibility is not always possible.

The goal is to remove avoidable ambiguity.

---

# 47. Guiding Rule

When measurement disagrees with theory:

> Trust the controlled measurement first, then investigate the theory.

A low-level optimization that appears obviously beneficial may still regress because of:

* VGPR pressure
* cache effects
* LDS pressure
* instruction scheduling
* graph topology
* launch behavior
* workload shape

MIInfer accepts optimizations based on measured behavior on the target hardware, not intuition alone.
