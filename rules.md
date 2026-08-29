Below is the `AGENTS.md` I would start MIInfer with. It is deliberately strict because Codex will otherwise tend to generalize the runtime, import too much from llama.cpp, or optimize without evidence.

````md
# AGENTS.md

## Project

**MIInfer** is an experimental, performance-first LLM inference runtime targeting **AMD gfx906**, initially the **AMD Instinct MI50 32GB**.

The project exists to answer a specific engineering question:

> How much inference performance can be extracted from gfx906 when the runtime, memory layout, execution plan, and kernels are designed specifically for this architecture rather than treating it as a legacy backend of a generic inference framework?

MIInfer is **not** intended to become another general-purpose llama.cpp, vLLM, PyTorch, or generic ROCm runtime.

Specialization is a feature.

---

# 1. Mission

The initial target is intentionally narrow:

- AMD Instinct MI50 32GB
- gfx906 / Vega20
- Linux
- single GPU
- batch 1 / low batch
- one explicitly selected model family
- one explicitly selected quantization path
- local inference
- reproducible benchmarking
- correctness before optimization
- measurable performance improvements over the strongest available gfx906 llama.cpp baseline

Portability is not an initial goal.

Do not broaden project scope without explicit approval.

---

# 2. Core hypothesis

We are testing:

## H0

A purpose-built gfx906 runtime cannot materially outperform a well-optimized llama.cpp/gfx906 implementation.

## H1

Removing generic-runtime constraints allows meaningful performance improvements on MI50/gfx906.

The project must be developed in a way that makes this hypothesis falsifiable.

Performance claims require reproducible evidence.

---

# 3. Fundamental engineering rules

These rules take precedence over convenience.

## 3.1 Measure before optimizing

Never assume that an optimization is useful because:

- another gfx906 project uses it
- it is theoretically faster
- it reduces instructions
- it increases occupancy
- it worked on another model
- it worked on another AMD GPU
- it worked in CUDA
- it worked in another MI50 workload

Profile first.

Benchmark the exact workload.

Then optimize the measured bottleneck.

---

## 3.2 Every optimization needs a baseline

An optimization without an A/B comparison is not an optimization.

For performance-sensitive changes record:

- baseline commit
- candidate commit
- model
- model hash when practical
- quantization
- context length
- batch size
- ROCm version
- compiler version
- GPU
- GPU clocks
- HBM clocks where available
- power limit
- temperature
- VRAM use
- benchmark command
- repeated measurements
- mean / median where appropriate
- correctness result
- performance delta

Do not report single-run wins as conclusions.

---

## 3.3 Negative results are valuable

Do not delete failed experiments merely because they failed.

Record them under the experiment system.

A useful negative result prevents future contributors from repeating the same work.

Example:

```text
Hypothesis:
Explicit Y-tile prefetch will hide HBM latency.

Result:
-7.4% PP.

Reason:
The workload already kept Y sufficiently L2-resident; extra loads increased
instruction and register pressure.

Decision:
REJECT for this workload.
````

---

# 4. Repository philosophy

MIInfer must remain architecturally independent.

Do not turn this repository into a llama.cpp fork.

External projects are:

* references
* correctness oracles
* benchmark competitors
* research sources
* sources of isolated ideas

They are not architectural authorities.

Relevant research projects currently include:

* ggml-org/llama.cpp
* milpster/gfx906-llama-cpp
* iacopPBK/llama.cpp-gfx906
* ai-infos/vllm-gfx906-mobydick
* Neroued/ninfer
* other gfx906-specific inference work discovered during research

Study them carefully, but understand the reason behind an optimization before transplanting it.

---

# 5. Architecture boundary

Commodity infrastructure may be reused when legally and technically sensible.

Examples:

* GGUF parsing concepts
* tokenizer handling
* model metadata definitions
* sampling algorithms
* quantization format definitions
* test vectors

The following are intended to remain MIInfer-owned architectural decisions:

* GPU memory layout
* packed weight representation
* gfx906 kernel library
* model execution planner
* static execution plans
* buffer lifetime planning
* activation reuse
* kernel selection
* graph capture strategy
* model specialization
* performance benchmark methodology

Do not introduce GGML's generic execution graph or scheduler simply to accelerate development.

That would invalidate an important part of the experiment.

---

# 6. Initial runtime model

Prefer:

```text
model
  ↓
load metadata
  ↓
validate supported architecture
  ↓
select exact MI50 kernels
  ↓
construct static execution plan
  ↓
allocate buffers
  ↓
load / repack weights
  ↓
capture reusable execution where appropriate
  ↓
decode
```

over:

```text
model
  ↓
generic graph
  ↓
dynamic operator selection
  ↓
generic scheduler
  ↓
generic backend
  ↓
HIP
```

Runtime decisions that can be made once during model loading should generally not be repeated for every token.

---

# 7. Initial hardware assumptions

The first hardware target is:

```text
AMD Instinct MI50
gfx906
Vega20
60 CUs
Wave64
32 GB HBM2
~1 TB/s memory bandwidth
```

Important architectural characteristics:

* no modern tensor-core/MFMA path equivalent to newer CDNA GPUs
* no rocWMMA strategy should be assumed
* Wave64 matters
* LDS usage matters
* VGPR pressure matters
* occupancy assumptions from RDNA/CDNA do not automatically transfer
* packed integer dot-product instructions may be useful
* bandwidth is often critical during low-batch decode

Do not treat `gfx906` as generic ROCm.

---

# 8. Precision policy

Do not assume one global inference dtype.

Precision is an operator-level decision.

Potential design space includes:

```text
weights:
Q4 / Q5 / Q6 / Q8 / MXFP4

activations:
FP16 / Q8

accumulation:
INT32 / FP32

normalization:
FP32 reduction where necessary

softmax:
FP32-sensitive operations where necessary

KV:
FP16
Q8 K + FP16 V
other formats only after measurement
```

Correctness takes priority over theoretical throughput.

Long-generation numerical stability must be tested.

---

# 9. Kernel philosophy

The kernel library should eventually expose reusable gfx906-native primitives.

Potential structure:

```text
gfx906/
├── primitives/
│   ├── dpp
│   ├── shuffle
│   ├── swizzle
│   ├── dot
│   ├── math
│   └── memory
│
├── gemv/
├── gemm/
├── quant/
├── norm/
├── rope/
├── attention/
└── moe/
```

Prefer explicit architecture-aware kernels over large generic template systems unless abstraction demonstrably has no performance cost.

---

# 10. Kernel experiments

Important research areas include:

## Matrix/vector execution

* Q4 × Q8
* Q6 × Q8
* Q8 × Q8
* MXFP4 where useful
* FP16 × FP16 → FP32
* half-wave cooperative execution
* Wave64 execution
* skinny GEMM
* GEMV
* MoE expert matrices

## Data layout

Compare:

```text
canonical model layout
vs
gfx906-native packed layout
```

Weight layout should match kernel consumption patterns where doing so produces measurable gains.

Avoid runtime repacking in hot paths.

---

## Activation reuse

Where several projections consume the same normalized activation:

```text
norm
 ↓
quantize once
 ↓
Q
K
V
```

is preferable to:

```text
norm
 ↓
quantize
Q

same norm
 ↓
quantize again
K
```

when correctness and profiling support the optimization.

Prefer static lifetime planning over dynamic cache discovery.

---

## Attention

Do not optimize attention simply because attention is conceptually important.

Profile first.

Test at several context lengths because bottlenecks shift with context.

At minimum consider:

* short context
* medium context
* long context

Potential experiments:

* FP16 KV
* Q8 K / FP16 V
* quantized KV only after quality validation

---

# 11. Static specialization

MIInfer is allowed to encode model-specific knowledge.

For an explicitly supported model, it is acceptable to know:

* hidden dimension
* head dimension
* number of heads
* number of layers
* expert count
* top-k
* intermediate dimensions
* quantization
* tensor layouts

Example:

```cpp
using Kernel = ExpertUpKernel<
    Gfx906,
    LogicalWave32,
    Fp32Accumulator,
    Mi50PackedQ4
>;
```

Avoid unnecessary runtime generality.

If adding generic behavior hurts hot-path performance or substantially complicates the architecture, stop and reassess.

---

# 12. Static kernel selection

Runtime autotuning is not automatically desirable.

For supported hardware/model combinations, prefer benchmark-derived configurations.

Conceptually:

```text
MI50
└── Model X
    ├── q_proj
    │   └── selected kernel configuration
    ├── k_proj
    ├── v_proj
    ├── attention
    └── experts
```

Autotuning may be used during development to discover configurations.

Production execution should eventually use known configurations where possible.

---

# 13. HIP graph strategy

Repeated decode steps are strong candidates for HIP graph capture/replay.

Do not implement graph capture before basic correctness.

Once execution is stable, measure:

```text
normal kernel dispatch
vs
captured/replayed decode
```

Graph topology itself should be treated as a performance parameter.

Small graph changes can have unexpectedly large performance effects.

---

# 14. Benchmarking

Benchmarks must distinguish at least:

```text
prompt processing / prefill
token generation / decode
```

Do not mix them into one headline number.

Useful benchmark dimensions include:

```text
PP512
PP2048
PP8192

TG128
TG512
TG1024
```

and context regimes such as:

```text
~512
~8K
~32K
```

The exact suite may evolve, but benchmarks must remain reproducible.

---

# 15. Hardware-state validation

Before trusting benchmark results, record hardware state.

At minimum when accessible:

* GPU model
* gfx target
* SCLK
* HBM/MCLK
* temperature
* power
* power cap
* VRAM allocated
* ROCm version
* compiler version
* kernel version

A benchmark run performed while GPU clocks are stuck or throttled is invalid.

Mark contaminated runs explicitly rather than silently deleting or publishing them.

---

# 16. Benchmark methodology

For small expected differences, use interleaved A/B testing.

Prefer:

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

to reduce machine-state drift.

For important claims:

* warm up first
* repeat runs
* reject obvious contaminated runs only with documented reason
* record raw data
* compute delta from aggregated results
* verify clocks and temperatures
* keep the baseline binary available

---

# 17. Correctness requirements

Performance regressions are bugs.

Correctness regressions are more serious bugs.

Kernel validation should progressively include:

### Level 1 — numerical

Compare against trusted CPU/reference implementation.

Track:

* max absolute error
* mean absolute error where useful
* relative error
* cosine similarity where meaningful

### Level 2 — layer/model

Compare:

* intermediate tensors where practical
* logits
* selected token

### Level 3 — generation

Test:

* short generations
* long generations
* repeated decoding
* long context

Watch for delayed numerical failures such as:

* NaN
* Inf
* repetitive garbage
* token divergence
* output collapse

A kernel that passes a short synthetic test is not automatically production-correct.

---

# 18. Experiment records

Every meaningful performance experiment should receive an ID.

Example:

```text
EXP-0001
EXP-0002
...
```

Suggested structure:

```text
experiments/
├── EXP-0001-q4-q8-gemv.md
├── EXP-0002-wave32-vs-wave64.md
├── EXP-0003-weight-repack.md
└── ...
```

Each experiment should contain:

```md
# EXP-NNNN — Title

## Hypothesis

## Motivation

## Baseline

## Candidate

## Environment

## Benchmark

## Correctness

## Results

## Profiling

## Interpretation

## Decision

KEEP / REJECT / RETEST

## Follow-up
```

Do not overwrite historical experiment conclusions after later discoveries.

Append a re-evaluation section instead.

---

# 19. Profiling

When possible, performance investigations should distinguish:

* compute bound
* HBM bandwidth bound
* cache bound
* instruction bound
* VGPR/register-pressure bound
* LDS bound
* launch/dispatch bound

Useful kernel-level metrics may include:

* kernel duration
* effective bandwidth
* occupancy
* VGPR count
* LDS use
* memory transactions
* cache behavior

Do not optimize occupancy as an isolated number.

For example, if LDS already limits a kernel to one resident workgroup per CU, reducing VGPR use solely to increase theoretical occupancy may provide no benefit.

---

# 20. External optimization intake

When examining another project, do not immediately port code.

Use this process:

```text
external optimization
       ↓
understand mechanism
       ↓
does it target gfx906?
       ↓
does it target our measured bottleneck?
       ↓
is the workload comparable?
       ↓
create isolated experiment
       ↓
correctness
       ↓
A/B benchmark
       ↓
KEEP / REJECT
```

Always distinguish:

```text
idea is good
```

from:

```text
idea is good for our exact workload
```

They are not equivalent.

---

# 21. Dependencies

Keep dependencies minimal.

Every major dependency should answer:

1. What problem does this solve?
2. Why should MIInfer not own this functionality?
3. Does it introduce runtime overhead?
4. Does it constrain gfx906 optimization?
5. Is its license compatible?
6. Does it pull in a generic execution architecture we are intentionally avoiding?

Avoid introducing PyTorch, vLLM, or another full inference runtime into the execution path.

---

# 22. Licensing and external code

Do not copy code from another repository without checking its license.

When adapting MIT-licensed code:

* retain required copyright notices
* retain applicable license text
* document substantial imported/adapted components

Prefer understanding and independently implementing small optimization concepts where practical.

Never remove attribution to make copied code appear original.

---

# 23. Scope control

Initial non-goals include:

* NVIDIA CUDA
* Intel GPU
* CPU inference optimization
* generic ROCm GPU support
* RDNA support
* MI200/MI300
* Windows
* macOS
* distributed inference
* tensor parallelism
* pipeline parallelism
* high-concurrency serving
* training
* fine-tuning
* arbitrary model support
* arbitrary quantization support
* multimodal inference unless explicitly selected later

Do not implement these without explicit approval.

---

# 24. Code quality

Performance code is still production code.

Requirements:

* clear ownership of buffers
* deterministic cleanup
* explicit error handling
* no silent fallback from GPU to CPU
* no hidden allocation in hot loops
* no unnecessary synchronization
* comments should explain architectural reasoning, not restate code
* assertions should validate assumptions that specialization relies upon

Example:

```cpp
static_assert(WAVE_SIZE == 64);
```

is preferable to silently supporting an architecture we have not validated.

Fail loudly for unsupported models or devices.

---

# 25. Hot-path rules

Avoid in token-generation hot paths:

* heap allocation
* string manipulation
* hash-map lookup
* dynamic graph analysis
* tensor-name lookup
* unnecessary virtual dispatch
* repeated shape inference
* repeated kernel selection
* repeated buffer planning

Prefer doing these once during load/planning.

---

# 26. Memory policy

VRAM is a first-class resource.

Track separately:

* weights
* KV cache
* persistent buffers
* activation buffers
* temporary workspace
* graph allocations
* optional optimization caches

An optimization that adds VRAM must justify its cost.

For example:

```text
+3% TG
+2 GB persistent VRAM
```

may be a bad trade if it materially reduces maximum context.

Do not judge speed independently of memory cost.

---

# 27. Performance metrics

Do not optimize only tokens/second.

Relevant metrics include:

* prompt tokens/s
* generation tokens/s
* time to first token
* VRAM
* context capacity
* watts
* tokens/joule
* latency consistency

When comparing implementations, ensure conditions are equivalent.

---

# 28. Reference baseline

The strongest working gfx906 llama.cpp configuration available to the project should be maintained separately as the primary external baseline.

MIInfer should not quietly compare itself only against stock llama.cpp if a materially faster gfx906-specialized implementation exists.

The benchmark opponent should be the strongest reproducible configuration we can establish.

---

# 29. Development sequence

Unless explicitly changed, follow this order:

## M0 — Baseline

* reproducible MI50 environment
* reference runtime
* target model
* benchmark suite
* hardware-state capture
* correctness oracle

## M1 — Kernel laboratory

* HIP build
* benchmark harness
* numerical reference
* first matvec/matmul kernels

## M2 — Prove specialization

* packed integer paths
* Wave64/half-wave experiments
* DPP/swizzle experiments
* layout experiments
* KEEP/REJECT evidence

M2 is a major go/no-go gate.

If specialization cannot produce meaningful kernel-level wins, reassess the project before building significant runtime infrastructure.

## M3 — Minimal runtime

Only after M2:

* model metadata
* tensor loading
* VRAM allocation
* static execution plan
* supported model validation

## M4 — First correct generation

Implement the minimum required for one model.

## M5 — Beat reference

Compare end-to-end against the chosen gfx906 baseline.

## M6 — Runtime specialization

Potential work:

* gfx906-native weight packing
* fused operations
* activation reuse
* fixed memory plan
* HIP graph capture

## M7 — Expansion

Only after the initial target succeeds:

* additional quantization
* second model
* long-context optimization
* speculative/MTP decoding
* optional server layer

---

# 30. Change discipline

For performance-sensitive changes:

1. State the hypothesis.
2. Identify expected bottleneck.
3. Establish baseline.
4. Implement smallest viable experiment.
5. Run correctness.
6. Benchmark.
7. Profile if results are unclear.
8. Record result.
9. Keep or revert based on evidence.

Avoid bundling unrelated optimizations into one commit.

A performance commit should ideally answer one question.

---

# 31. Git discipline

Prefer focused commits.

Examples:

```text
bench: establish q4_q8 gemv baseline

gfx906: add wave32 cooperative q4_q8 prototype

bench: record wave32 gemv results

gfx906: reject explicit y prefetch experiment
```

Do not combine:

```text
new kernel
+ new allocator
+ formatting
+ tokenizer rewrite
+ benchmark changes
```

into one commit.

Performance history must remain bisectable.

---

# 32. Codex task behavior

When working on a task:

### Before modifying code

Inspect:

* current architecture
* relevant tests
* benchmark harness
* experiment history
* recent related commits

Do not assume an optimization has not already been tested.

### Before adding abstractions

Ask internally whether specialization makes the abstraction unnecessary.

Prefer the smallest architecture consistent with the supported target.

### Before adding fallback behavior

Do not silently support unsupported hardware or models.

Return an explicit error.

### Before declaring a performance win

Provide evidence.

### Before removing apparently strange gfx906 code

Find out why it exists.

Hardware-specific code often encodes measured constraints that are not obvious from generic programming principles.

---

# 33. Definition of done

A normal functional change is done when:

* it builds
* tests pass
* unsupported conditions fail clearly
* documentation is updated where necessary
* no accidental scope expansion occurred

A performance change is done only when:

* correctness passes
* benchmark methodology is documented
* baseline and candidate are comparable
* repeated measurements exist
* hardware state is valid
* result is recorded
* performance and VRAM tradeoffs are understood
* KEEP / REJECT decision is explicit

---

# 34. Guiding principle

When choosing between:

```text
generic and elegant
```

and:

```text
simple, explicit, measured, and faster on MI50
```

this project usually chooses the second.

But specialization without measurement is cargo culting.

The governing loop is:

```text
UNDERSTAND
   ↓
MEASURE
   ↓
HYPOTHESIZE
   ↓
IMPLEMENT
   ↓
VERIFY CORRECTNESS
   ↓
BENCHMARK
   ↓
KEEP OR REJECT
```

MIInfer succeeds only if its performance claims survive that loop.

````

### One recommendation before you give Codex its first implementation task

I would make the first Codex task **repository/bootstrap + M0/M1 infrastructure**, not “implement an LLM runtime.”

The first meaningful tree should probably end up around:

```text
MIInfer/
├── AGENTS.md
├── README.md
├── CMakeLists.txt
├── cmake/
├── src/
├── include/
├── gfx906/
│   ├── primitives/
│   └── kernels/
├── bench/
├── tests/
├── experiments/
└── docs/
    ├── architecture.md
    ├── benchmarking.md
    └── research/
````

And I would make the **first real technical milestone a standalone gfx906 Q4/Q8 GEMV benchmark with a CPU correctness oracle**, before model loading, tokenizers, servers, or any other runtime machinery. That gives us the quickest possible answer to whether the central MIInfer hypothesis has legs.
