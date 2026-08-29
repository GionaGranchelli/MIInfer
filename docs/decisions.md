# MIInfer Architectural Decisions

This document records major project decisions that should not be casually reopened during implementation.

These are not permanent laws.

A decision may be changed when new evidence invalidates the assumptions behind it, but the change should be explicit and documented.

---

# D001 — Build a New Runtime Instead of Forking llama.cpp

**Status:** Accepted

## Decision

MIInfer will be developed as a new project rather than as a long-lived fork of llama.cpp.

## Reason

The purpose of MIInfer is to test whether removing generic-runtime constraints enables meaningful performance improvements on gfx906.

Building directly on llama.cpp would inherit major architectural assumptions including:

* GGML execution graph
* generic backend architecture
* generic model support
* dynamic scheduling decisions
* portability constraints
* upstream architectural churn

Those constraints are precisely part of what MIInfer is trying to avoid.

## Consequences

MIInfer may still use llama.cpp as:

* a correctness reference
* a performance baseline
* a model-format reference
* a source of implementation knowledge
* a source of isolated reusable components where licensing permits

MIInfer should not adopt llama.cpp's execution architecture by default.

---

# D002 — Maintain a Separate gfx906 Reference Runtime

**Status:** Accepted

## Decision

A separate optimized gfx906 llama.cpp implementation will be maintained outside the MIInfer runtime as the primary external reference.

## Reason

MIInfer needs a strong opponent.

Comparing only against stock llama.cpp would be misleading if better gfx906-specialized implementations already exist.

## Consequences

The reference runtime should be:

* pinned to exact commits
* built reproducibly
* benchmarked under equivalent conditions
* kept architecturally separate from MIInfer

The reference implementation should not become a runtime dependency.

---

# D003 — MI50 32GB / gfx906 Is the Initial Hardware Contract

**Status:** Accepted

## Decision

The initial supported hardware target is:

```text
AMD Instinct MI50 32GB
gfx906 / Vega20
single GPU
Linux
```

## Reason

Hardware specialization is the core experiment.

Supporting multiple architectures early would weaken the ability to make aggressive gfx906-specific decisions.

## Consequences

MIInfer may assume:

* Wave64
* gfx906 instruction behavior
* MI50 memory characteristics
* single-device execution

Unsupported hardware should fail explicitly.

---

# D004 — Portability Is Not an Initial Goal

**Status:** Accepted

## Decision

MIInfer will not initially optimize for:

* NVIDIA
* Intel GPU
* RDNA
* newer CDNA
* CPU inference
* Windows
* macOS

## Reason

Generic portability introduces abstraction and testing costs that do not help answer the initial project hypothesis.

## Consequences

Architecture-specific code is acceptable.

Conditional branches for hypothetical future hardware should not be added without a current need.

---

# D005 — Performance Claims Require Controlled Measurement

**Status:** Accepted

## Decision

No performance optimization is accepted based solely on intuition, theory, profiler metrics, or external reports.

Important performance changes require controlled A/B measurement.

## Required evidence

Where appropriate:

* exact baseline
* exact candidate
* repeated measurements
* comparable hardware state
* correctness
* environment capture
* interpretation
* explicit KEEP / REJECT / RETEST decision

## Reason

gfx906 optimization results are highly workload-sensitive.

Optimizations that worked in other projects have sometimes regressed substantially in different workloads.

---

# D006 — Negative Experiments Are Retained

**Status:** Accepted

## Decision

Meaningful failed optimization experiments should be documented rather than erased from project history.

## Reason

Negative results are useful project knowledge.

They help avoid repeating work and reveal workload-specific behavior.

## Consequences

Experiment records should remain available after rejection.

Later evidence may cause a rejected experiment to be revisited, but the original result should remain preserved.

---

# D007 — M2 Is a Go/No-Go Gate

**Status:** Accepted

## Decision

Significant runtime implementation should not proceed automatically before the project demonstrates credible gfx906 specialization wins.

## Reason

A complete inference runtime is expensive to build.

The project should first prove that architecture-specific kernels have enough performance potential to justify that investment.

## Consequences

M0/M1/M2 prioritize:

* environment
* benchmark infrastructure
* correctness
* kernel experiments

If M2 does not demonstrate meaningful potential, the project should be reassessed.

---

# D008 — Do Not Build a Generic Graph Runtime Initially

**Status:** Accepted

## Decision

MIInfer will not begin with a generic tensor/operator/graph/scheduler architecture.

## Reason

The initial model architecture, hardware, shapes, and supported quantization are known.

Many runtime decisions can therefore be made once during planning rather than repeatedly during execution.

## Preferred direction

```text
model
  ↓
validate
  ↓
plan
  ↓
allocate
  ↓
select kernels
  ↓
execute fixed plan
```

## Consequences

Do not introduce a generic computational graph merely for architectural familiarity.

---

# D009 — Static Decisions Belong Outside the Decode Hot Path

**Status:** Accepted

## Decision

Information known at model load should generally not be rediscovered during token generation.

## Examples

Prefer performing once:

* shape validation
* kernel selection
* buffer planning
* tensor lookup
* quantization-path selection
* launch configuration selection

## Reason

Decode is repetitive and latency-sensitive.

Repeated dynamic decisions add overhead without providing value for a fixed model/hardware combination.

---

# D010 — No Silent CPU or Generic Fallback

**Status:** Accepted

## Decision

Unsupported configurations should fail clearly.

MIInfer should not silently:

* offload unsupported operators to CPU
* select a generic ROCm implementation
* change model dtype
* change quantization
* reduce context
* alter execution semantics

## Reason

Silent fallback makes performance and correctness difficult to reason about.

Specialization should remain explicit.

---

# D011 — Use Per-Operation Precision

**Status:** Accepted

## Decision

MIInfer will not assume a single global dtype for inference.

Precision may differ between:

* weights
* activations
* accumulators
* output
* KV cache
* softmax/reductions

## Example

```text
weight: Q4
activation: Q8
dot accumulation: INT32
scale/reduction: FP32
output: FP16
```

## Reason

gfx906 precision behavior and numerical requirements differ by operation.

---

# D012 — FP32 Accumulation Is Allowed Where Correctness Requires It

**Status:** Accepted

## Decision

MIInfer may use FP32 or INT32 accumulation even when inputs are lower precision.

## Reason

The fastest low-precision path is not useful if it causes numerical instability.

Some gfx906 community work has demonstrated that accumulator precision can materially affect correctness.

## Consequences

Accumulator type must be treated as an explicit kernel parameter.

---

# D013 — BF16 Is Not a Default Internal Format

**Status:** Accepted

## Decision

MIInfer will not treat BF16 as the default floating-point execution type.

## Reason

gfx906 does not provide the same native BF16 behavior as modern accelerators.

Using BF16 blindly can result in inefficient fallback behavior.

## Initial preference

Where floating-point execution is needed:

```text
FP16 operands
FP32 accumulation where necessary
```

BF16 may be evaluated later if evidence supports it.

---

# D014 — Kernel-Native Weight Layout Is Allowed

**Status:** Accepted

## Decision

MIInfer is allowed to transform model weights into a representation designed specifically for gfx906 kernels.

## Reason

External model formats are designed for portability and storage, not necessarily optimal MI50 execution.

Weight repacking may reduce:

* addressing complexity
* scale lookup overhead
* uncoalesced loads
* unpacking cost

## Consequences

External representation and internal representation are separate concepts.

Repacking must still justify its conversion cost and VRAM/storage implications.

---

# D015 — Repacking Should Not Occur in Hot Paths

**Status:** Accepted

## Decision

If weight repacking is beneficial, it should happen:

* offline
* during conversion
* or once during model loading

It should not occur during token generation.

## Reason

Decode should operate directly on execution-ready weights.

---

# D016 — Static Activation Reuse Is Preferred Over Dynamic Cache Discovery

**Status:** Accepted

## Decision

When multiple operators consume the same transformed or quantized activation, MIInfer should prefer explicit planner-managed reuse.

## Example

```text
RMSNorm
   ↓
Q8 activation
   ├── Q
   ├── K
   └── V
```

## Reason

For a fixed execution graph, consumer relationships are already known.

Dynamic caches based on tensor identities, names, generations, or runtime graph scanning add unnecessary complexity.

---

# D017 — Prefill and Decode May Use Different Kernels

**Status:** Accepted

## Decision

MIInfer will not force one implementation to handle both prefill and decode optimally.

## Reason

They are structurally different workloads.

Prefill often resembles larger matrix multiplication.

Decode often resembles GEMV/skinny GEMM and is much more sensitive to memory bandwidth and launch overhead.

## Consequences

Separate optimized paths are acceptable and expected.

---

# D018 — Attention Is Not Automatically the First Optimization Target

**Status:** Accepted

## Decision

MIInfer will not begin by writing a complex FlashAttention implementation unless profiling demonstrates that attention is a major current bottleneck.

## Reason

Existing gfx906 profiling has shown workloads where matrix multiplication dominates while attention contributes only a small fraction of execution time.

## Consequences

Profile the target model first.

Optimize the measured bottleneck.

---

# D019 — Long Context Is a Distinct Performance Regime

**Status:** Accepted

## Decision

Short-context decode results must not be assumed to represent long-context behavior.

## Reason

As context grows, KV traffic and attention can become increasingly important.

## Consequences

Important end-to-end benchmarks should eventually test multiple context regimes.

---

# D020 — HIP Graph Capture Comes After Correct Execution

**Status:** Accepted

## Decision

HIP graph capture is a planned optimization, not an initial execution requirement.

## Order

```text
correct execution
      ↓
profile dispatch overhead
      ↓
stable memory addresses
      ↓
stable graph topology
      ↓
capture/replay experiment
```

## Reason

Graph capture adds complexity and can make debugging harder.

Its benefit must be measured.

---

# D021 — Graph Topology Is a Performance Parameter

**Status:** Accepted

## Decision

Changes to execution graph shape must be benchmarked even when they appear semantically equivalent.

## Reason

Existing gfx906 work has demonstrated large performance changes caused by apparently small graph-shape modifications.

## Consequences

Do not dismiss measured graph regressions because source-level reasoning suggests the change “should not matter.”

---

# D022 — Dependencies Must Remain Minimal

**Status:** Accepted

## Decision

MIInfer should not introduce large framework dependencies without strong justification.

## Initially disallowed by default

* PyTorch runtime
* vLLM
* GGML execution framework
* generic accelerator frameworks

## Reason

Dependencies can:

* constrain architecture
* introduce hidden overhead
* complicate deployment
* weaken control of the performance-critical path

Small focused dependencies remain acceptable where justified.

---

# D023 — Triton Is a Research Tool, Not a Required Runtime

**Status:** Accepted

## Decision

A gfx906-compatible Triton stack may be used to prototype kernels where useful.

It should not automatically become a production dependency.

## Preferred role

```text
idea
 ↓
Triton prototype
 ↓
correctness
 ↓
benchmark
 ↓
profile
 ↓
HIP/ISA implementation if justified
```

---

# D024 — Benchmarkability Is an Architectural Requirement

**Status:** Accepted

## Decision

Performance-critical kernels should be independently callable where practical.

## Reason

The same implementation should support:

* unit/reference testing
* microbenchmarking
* profiling
* model execution

This prevents the full runtime from becoming a prerequisite for every performance investigation.

---

# D025 — VRAM Is Part of Optimization Cost

**Status:** Accepted

## Decision

Performance improvements must be evaluated together with memory cost.

## Example

```text
+3% generation throughput
+2 GB persistent VRAM
```

is not automatically a good optimization.

## Reason

Additional persistent VRAM reduces:

* context capacity
* supported model size
* workspace margin

Throughput alone is insufficient.

---

# D026 — Strongest Available Relevant Baseline Wins

**Status:** Accepted

## Decision

MIInfer should compare itself against the strongest reproducible relevant gfx906 path available, not merely the easiest baseline to beat.

## Reason

The project is intended to answer a real engineering question rather than produce favorable benchmark marketing.

---

# D027 — Serving Does Not Define the Core Runtime

**Status:** Accepted

## Decision

CLI, HTTP, and OpenAI-compatible serving belong outside the core execution architecture.

## Reason

Serving concerns should not dictate:

* memory layout
* kernel architecture
* execution graph
* scheduling behavior

before the single-model execution engine has demonstrated value.

---

# D028 — One Model First

**Status:** Accepted

## Decision

The first runtime implementation will support one explicitly chosen model family/configuration.

## Reason

Supporting many model architectures early would force generic abstractions before the project understands which abstractions are actually useful.

## Consequences

Unsupported architectures should fail explicitly.

A second model becomes valuable later as a test of whether the architecture can generalize without becoming generic.

---

# D029 — Batch 1 / Low Batch Is the Initial Optimization Target

**Status:** Accepted

## Decision

MIInfer initially optimizes for interactive single-user inference rather than high-concurrency serving.

## Reason

Batch-1 decode emphasizes exactly the gfx906 problems MIInfer intends to investigate:

* HBM traffic
* GEMV
* skinny GEMM
* launch overhead
* static execution

## Consequences

Do not optimize architecture around continuous batching or large request queues during early milestones.

---

# D030 — The Project May End After M2 or M5

**Status:** Accepted

## Decision

MIInfer is allowed to conclude that the specialization hypothesis does not justify a standalone runtime.

## Reason

The project is an engineering investigation, not a commitment to produce a framework regardless of evidence.

## Possible valid outcomes

* M2 shows insufficient kernel advantage
* M5 shows kernel gains disappear end-to-end
* complexity outweighs benefits
* quality/memory trade-offs are unacceptable

Any of these may justify stopping or changing direction.

That result should be documented rather than treated as failure.

---

# Decision Change Process

A major accepted decision may be changed when evidence justifies it.

When changing a decision:

1. do not delete the original rationale
2. mark the old decision as `Superseded`
3. add the new decision
4. link the relevant experiments or measurements
5. update `architecture.md`
6. update `roadmap.md` if scope changes
7. update `current-state.md`

Example:

```text
D014 — Kernel-native weight layout
Status: Superseded by D041
```

The purpose of this file is to preserve reasoning over time.

---

# Guiding Rule

Do not reopen accepted architectural decisions merely because a different design would be more conventional.

Reopen them when:

> new evidence shows that an assumption behind the decision is wrong or that another approach materially improves the project's ability to answer its core performance question.
