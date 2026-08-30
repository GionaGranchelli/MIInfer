# MIInfer Architecture

This document defines the architectural direction and boundaries of MIInfer.

It is intentionally high-level.

The goal is to constrain implementation decisions enough to preserve the project's purpose while leaving room for benchmark-driven design changes.

---

# 1. Architectural Objective

MIInfer is a specialized inference runtime for **AMD gfx906**, initially the **AMD Instinct MI50 32GB**.

The architecture should optimize for:

* one known GPU architecture
* one known execution environment
* one explicitly supported model family
* low-batch inference
* static execution wherever possible
* predictable memory use
* architecture-specific kernels
* reproducible performance

The architecture should not optimize for broad portability.

---

# 2. Core Design Question

The central architectural question is:

> What does an LLM inference runtime look like if gfx906 is treated as the primary hardware target rather than as one backend among many?

This leads to a fundamentally different design from generic inference frameworks.

Generic runtimes typically need to support:

```text
many models
many hardware targets
many dtypes
many schedulers
many batch patterns
many execution graphs
```

MIInfer initially does not.

That specialization should be used deliberately.

---

# 3. Primary Execution Model

The intended execution flow is:

```text
model artifact
      │
      ▼
model metadata
      │
      ▼
supported configuration validation
      │
      ▼
tensor loading / conversion / repacking
      │
      ▼
static memory planning
      │
      ▼
static kernel selection
      │
      ▼
execution plan
      │
      ▼
prefill / decode
      │
      ▼
gfx906 kernels
      │
      ▼
MI50
```

The execution plan should be created once whenever possible.

Hot-path execution should not rediscover information already known at model load.

M3 implements this boundary for the pinned Qwen3-8B Q4_0 artifact: a
mmap-backed, fail-closed loader produces strongly named tensor handles, one
contiguous GPU weight arena, and an immutable static projection-kernel plan.
The plan is validation and ownership infrastructure only; transformer
execution remains a later milestone.

M4-A adds correctness-first primitive ownership below that boundary. Host
oracles define the Qwen3 math contract, while gfx906 code owns embedding
lookup, RMSNorm, Q6_K output-projection arithmetic, and the layer-0
composition path. The layer-0 executor and its four-position KV-cache state
are validated against an independent pinned reference trace; full 36-layer
execution remains an M4-B concern.

---

# 4. Architectural Non-Goal: Generic Graph Runtime

MIInfer should not begin with a generic computational graph abstraction.

Avoid introducing something conceptually equivalent to:

```text
Tensor
  ↓
Operator
  ↓
Graph
  ↓
Graph Optimizer
  ↓
Scheduler
  ↓
Backend
```

unless future evidence demonstrates that such machinery is necessary.

The initial model graph is known.

The GPU architecture is known.

The tensor dimensions are known.

The quantization format is known.

Therefore much of the work generic graph systems perform dynamically can be decided statically.

---

# 5. Runtime Layers

The architecture is expected to evolve around a small number of conceptual layers.

These are responsibilities, not mandatory C++ class names.

---

## 5.1 Model Layer

Responsibilities:

* read model metadata
* identify architecture
* validate supported configuration
* enumerate required tensors
* validate tensor shapes
* expose model dimensions to the planner

The model layer should not perform dynamic execution scheduling.

Unsupported configurations should fail explicitly.

---

## 5.2 Packing / Representation Layer

Responsibilities:

* transform external tensor representations into kernel-native representations where useful
* maintain alignment requirements
* define scale/value layouts
* preserve reproducibility of packed formats

Potential transformations include:

```text
canonical GGUF quant layout
        ↓
MI50-native packed layout
```

Packing may happen:

* offline
* at model load
* through a dedicated converter

The correct strategy should be determined experimentally.

Runtime hot-path repacking should be avoided.

---

## 5.3 Memory Planner

Responsibilities:

* weight residency
* activation buffers
* KV cache
* temporary workspace
* reusable scratch buffers
* graph-capture-compatible storage where necessary

The planner should know lifetimes before execution where possible.

Prefer:

```text
plan once
allocate once
reuse
```

over repeated allocation.

---

## 5.4 Kernel Planner

Responsibilities:

* choose exact kernel implementation
* select launch configuration
* select tensor representation
* select precision policy
* encode model-specific shape decisions

Example conceptually:

```text
q_proj:
  q4_q8_gemv_v3
  logical_width = 32
  workgroup = 256
  accumulation = fp32
```

Kernel selection should normally happen at load/planning time rather than every token.

---

## 5.5 Execution Plan

The execution plan binds together:

* tensors
* buffers
* kernel choices
* operation order
* synchronization requirements

Conceptually:

```text
Layer 0
  RMSNorm
  Q projection
  K projection
  V projection
  RoPE
  Attention
  Output projection
  Residual
  FFN/MoE
  Residual

Layer 1
  ...
```

For supported models, execution plans may encode known architectural structure directly.

---

## 5.6 Kernel Layer

The kernel layer contains gfx906-specific implementations.

Expected areas include:

```text
gfx906/
├── primitives/
├── quant/
├── gemv/
├── gemm/
├── norm/
├── rope/
├── attention/
└── moe/
```

Kernels may use architecture-specific assumptions such as:

* Wave64
* DPP
* ds_swizzle
* packed dot products
* explicit LDS layouts
* specific VGPR/occupancy trade-offs

Do not hide gfx906-specific behavior behind unnecessary generic abstractions.

---

# 6. gfx906 Primitive Layer

MIInfer should develop a reusable architecture-native primitive layer.

Possible categories:

```text
gfx906/primitives/
├── lane
├── dpp
├── swizzle
├── reduction
├── dot
├── math
└── memory
```

Examples of useful functionality may include:

* lane broadcast
* Wave64 reduction
* logical half-wave reduction
* DPP shuffle/reduction
* DS swizzle
* packed integer dot operations
* architecture-native reciprocal/exp operations where justified

The purpose is to avoid duplicating low-level ISA logic across kernels.

---

# 7. Wave64 and Logical Cooperative Width

gfx906 uses Wave64.

This does not imply that every logical operation must treat all 64 lanes as one cooperative group.

Potential execution modes include:

```text
64-lane logical group

2 × 32-lane logical groups

4 × 16-lane logical groups
```

The best mapping depends on workload shape.

Do not encode a universal assumption that Wave64 implies one row per wave.

Benchmark representative model shapes.

---

# 8. Precision Architecture

MIInfer should represent precision choices explicitly.

Do not assume a single dtype determines an operation.

Conceptually:

```text
Kernel<
    WeightType,
    ActivationType,
    AccumulatorType,
    OutputType
>
```

Possible examples:

```text
Q4 weight
Q8 activation
INT32 dot accumulation
FP32 scale accumulation
FP16 output
```

or:

```text
FP16 weight
FP16 activation
FP32 accumulation
FP16 output
```

Precision should be chosen per operation.

---

# 9. Quantized Activation Reuse

Some model stages have multiple consumers of the same normalized activation.

For example:

```text
                ┌── Q
RMSNorm ── Q8 ──┼── K
                └── V
```

or:

```text
                    ┌── router
FFN RMSNorm ── Q8 ──┼── gate
                    └── up
```

Where beneficial, quantize once and reuse.

Prefer static planning over a dynamic activation-cache lookup system.

The planner knows the graph and consumer relationships in advance.

---

# 10. Memory Architecture

VRAM is treated as a constrained resource.

Memory should be categorized explicitly:

```text
weights
KV cache
persistent transformed weights
persistent activations
temporary activations
workspace
graph-capture allocations
optional caches
```

Every persistent optimization has a memory cost.

Performance improvements must be evaluated against reduced context capacity or other VRAM trade-offs.

---

# 11. Weight Residency

The initial target assumes the supported model fits entirely within a single MI50.

Prefer full GPU residency.

Do not introduce CPU offload architecture unless explicitly required later.

The initial runtime should fail if the selected supported configuration cannot fit rather than silently introducing complex offloading behavior.

---

# 12. Tensor Layout

External model layout and execution layout are separate concepts.

MIInfer is free to store weights differently from GGUF/Hugging Face representations.

For example:

```text
external:
[block][quant bytes][scale]

internal:
[all quant bytes plane]
[all scales plane]
```

or another kernel-friendly arrangement.

The exact format must be benchmark driven.

Custom layout is one of the project's intended areas of differentiation.

---

# 13. Prefill vs Decode

Prefill and decode are distinct workloads.

Do not assume the same kernels or execution strategies are optimal for both.

Typical characteristics:

## Prefill

* larger M dimension
* more parallel work
* potentially compute-heavy
* larger tiles may be useful

## Decode

* very small M
* bandwidth sensitive
* launch overhead matters more
* GEMV/skinny GEMM characteristics dominate

Architectural decisions should allow separate optimized paths.

---

# 14. Context-Length Sensitivity

The dominant bottleneck can change with context.

Conceptually:

```text
short context
  ↓
weight/model execution dominates

long context
  ↓
attention/KV traffic becomes increasingly important
```

Therefore benchmark and kernel selection may eventually depend on context regime.

Do not assume short-context results extrapolate to long context.

---

# 15. Attention Architecture

Attention should initially favor correctness and simplicity.

Possible progression:

```text
correct reference attention
        ↓
profile
        ↓
optimized gfx906 attention
        ↓
KV compression / quantization experiments
```

Do not begin the project by writing a complex FlashAttention implementation unless profiling proves attention is the primary early bottleneck.

Potential future KV configurations include:

```text
FP16 K / FP16 V

Q8 K / FP16 V

other combinations only after correctness and quality testing
```

---

# 16. MoE Architecture

If the first target model is MoE, MIInfer may specialize heavily for its exact expert structure.

Potential static knowledge includes:

* number of experts
* active experts per token
* expert dimensions
* routing top-k
* fixed expert weight layouts

Potential execution flow:

```text
normalized activation
        ↓
router
        ↓
top-k selection
        ↓
expert dispatch
        ↓
gate/up projection
        ↓
activation
        ↓
down projection
        ↓
weighted combine
```

Avoid generic expert scheduling designed for arbitrary concurrency if batch-1 inference does not require it.

---

# 17. Static Model Knowledge

Model-specific constants may be compiled or generated into configuration.

Examples:

```text
hidden_size
head_dim
num_heads
num_kv_heads
intermediate_size
num_layers
num_experts
top_k
```

This is acceptable.

The architecture should not pretend these values are unknown when supporting one explicit model.

---

# 18. Kernel Configuration

Kernel configuration can be discovered experimentally and then stored statically.

Potential development process:

```text
enumerate candidate configurations
        ↓
benchmark exact model shapes
        ↓
select winner
        ↓
record result
        ↓
ship selected config
```

Runtime autotuning should not be added unless it provides clear value.

---

# 19. HIP Graph Capture

Decode execution is repetitive and may benefit from HIP graph capture.

Potential flow:

```text
construct stable decode execution
        ↓
capture
        ↓
replay token 1
replay token 2
replay token 3
...
```

Graph capture is not an initial correctness requirement.

It should be added only after:

* execution is stable
* memory addresses are stable
* kernel topology is understood
* ordinary dispatch performance is measured

Graph structure itself must be benchmarked.

---

# 20. Synchronization Philosophy

Avoid unnecessary host/device synchronization.

Where possible:

* keep execution GPU-resident
* use asynchronous launches
* synchronize only when required
* avoid device-wide synchronization between operations

Correctness and benchmark instrumentation may temporarily require additional synchronization during development.

Hot-path production execution should minimize it.

---

# 21. Allocation Philosophy

Hot paths should not allocate.

Avoid token-by-token:

* `new`
* `malloc`
* `hipMalloc`
* dynamic vector growth
* hash-map insertion
* string allocation

Persistent and scratch memory should be planned beforehand.

---

# 22. Error Handling

Specialization should fail loudly.

Examples:

```text
unsupported GPU
unsupported model architecture
wrong tensor dimension
unsupported quantization
insufficient VRAM
invalid packed artifact
```

should produce clear errors.

Do not silently:

* fall back to CPU
* select generic ROCm code
* change quantization
* change model behavior

---

# 23. External Code Boundary

External implementations may be used as references.

Potentially reusable commodity components include:

* GGUF parsing
* tokenizer logic
* model metadata structures
* sampling implementations
* quantization format definitions

The architectural core should remain independent.

Do not import:

* GGML graph execution
* GGML scheduler
* generic backend abstraction
* vLLM scheduling architecture

merely to reduce implementation time.

---

# 24. Reference Runtime Relationship

A separate optimized gfx906 llama.cpp implementation should be maintained as an external reference.

It serves as:

```text
performance baseline
correctness oracle
feature reference
regression comparison
```

MIInfer should not depend on it at runtime.

---

# 25. Initial Source Layout

The current intended source structure is:

```text
include/
└── miinfer/

src/
├── model/
├── memory/
├── runtime/
└── sampling/

gfx906/
├── primitives/
├── quant/
├── gemv/
├── gemm/
├── norm/
├── rope/
├── attention/
└── moe/

bench/
tests/
experiments/
```

This layout is provisional.

Do not create empty abstractions purely to match this directory structure.

Implementation evidence may change it.

---

# 26. Dependency Direction

Preferred dependency direction:

```text
model metadata
      ↓
planner
      ↓
runtime
      ↓
gfx906 kernels
```

Kernel code should not depend on model-loading infrastructure.

Benchmark code should be able to call kernels independently of the full runtime.

This separation is important for M1/M2 experimentation.

---

# 27. Benchmarkability as an Architectural Requirement

Every important kernel should be callable independently where practical.

This allows:

```text
microbenchmark
correctness test
profiling
model execution
```

to exercise the same implementation.

Avoid hiding performance-critical kernels behind interfaces that make isolated benchmarking difficult.

---

# 28. Testability

The architecture should permit CPU/reference comparisons without requiring the complete runtime.

Examples:

```text
reference Q4×Q8 GEMV
vs
gfx906 Q4×Q8 GEMV
```

or:

```text
reference RMSNorm
vs
gfx906 RMSNorm
```

Tests should not depend on model serving.

---

# 29. Generated Configuration

If benchmark-driven tuning produces large shape-specific configuration tables, generated source/configuration is acceptable.

Example:

```text
generated/
└── gfx906/
    └── qwen-target/
        ├── gemv.hpp
        ├── attention.hpp
        └── moe.hpp
```

Generated files should include:

* source benchmark data reference
* generator version
* target architecture
* generation date or commit

Avoid opaque magic numbers with no provenance.

---

# 30. Future Packed Artifact

MIInfer may eventually introduce its own packed artifact format.

Conceptually:

```text
Hugging Face / GGUF
        ↓
MIInfer converter
        ↓
.mi50 artifact
        ↓
direct MI50-native load
```

Potential contents:

* metadata
* tokenizer data
* prepacked weights
* alignment information
* format version
* model checksum

This is not required in early milestones.

First prove that repacking matters.

---

# 31. CLI / Serving Boundary

CLI and serving are outside the core runtime.

Eventually:

```text
CLI
  │
  ├── runtime
HTTP ─┘
```

The runtime must not depend on HTTP/server concepts.

Serving concerns should not dictate GPU execution architecture.

---

# 32. Architectural Decision Rule

When deciding whether to add a runtime abstraction, ask:

1. Is this required by the current supported target?
2. Does it exist only to support hypothetical future hardware/models?
3. Does it add hot-path overhead?
4. Does it make isolated kernel benchmarking harder?
5. Can the decision instead happen once at load time?
6. Is there benchmark evidence that the abstraction is useful?

If the answer is primarily future flexibility, defer it.

---

# 33. Architecture Evolution

This document is not immutable.

Architecture may change when experiments demonstrate that an assumption is wrong.

When changing a major architectural direction:

* record the evidence
* update `docs/decisions.md`
* update this document
* update `docs/current-state.md`
* preserve relevant experiment history

Architecture changes should be evidence-driven rather than convenience-driven.

---

# 34. Guiding Architecture Principle

The intended runtime is closer to:

```text
known model
+
known GPU
+
known tensor layouts
+
known memory plan
+
known kernels
=
small explicit execution engine
```

than:

```text
arbitrary model
+
arbitrary graph
+
arbitrary hardware
+
dynamic scheduler
=
general inference framework
```

MIInfer exists specifically to explore the performance implications of that difference.
