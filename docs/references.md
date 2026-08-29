# MIInfer Research References

This document records external projects, implementations, and research that are relevant to MIInfer.

These projects are used as:

* research sources
* performance references
* correctness references
* sources of optimization hypotheses
* records of known gfx906 limitations

They are **not automatically dependencies**.

An optimization appearing in another project is not evidence that it will improve MIInfer.

External ideas must pass MIInfer's normal process:

```text
understand
   ↓
bottleneck match
   ↓
isolated experiment
   ↓
correctness
   ↓
benchmark
   ↓
profile
   ↓
KEEP / REJECT
```

---

# 1. ggml-org/llama.cpp

**Repository:** `ggml-org/llama.cpp`

## Role

Primary general-purpose reference implementation.

Use llama.cpp as:

* model-behavior reference
* GGUF reference
* quantization-format reference
* tokenizer reference
* correctness oracle
* benchmark comparison
* source of implementation knowledge

MIInfer is intentionally **not** a llama.cpp fork.

## Important lessons

llama.cpp demonstrates the engineering required to support:

* many model architectures
* many quantization formats
* multiple hardware backends
* CPU/GPU execution
* model conversion
* tokenization
* sampling
* serving
* multi-GPU execution

Those capabilities are useful references but impose generic-runtime constraints MIInfer is deliberately attempting to avoid.

## Potentially reusable knowledge

Study:

* GGUF structures
* quantization definitions
* model metadata
* tokenizer behavior
* sampling behavior
* reference tensor operations
* model architecture implementations

## Do not inherit automatically

Do not adopt merely for convenience:

* GGML execution graph
* generic scheduler
* generic backend abstraction
* generic tensor/operator dispatch
* dynamic model execution architecture

## MIInfer question

For every llama.cpp mechanism, ask:

> Is this complexity necessary because inference requires it, or because llama.cpp supports many configurations that MIInfer does not?

---

# 2. milpster/gfx906-llama-cpp

**Repository:** `milpster/gfx906-llama-cpp`

## Role

Current high-value gfx906 research and reference project.

This project is especially important for MIInfer because of its **engineering methodology**.

It should be considered a strong candidate for the external gfx906 llama.cpp baseline.

## Important lessons

The project demonstrates disciplined performance engineering through:

* exact baselines
* controlled experiments
* hardware/software configuration records
* interleaved A/B benchmarking
* negative-result preservation
* optimization reverts
* contamination detection
* upstream synchronization
* explicit KEEP / REJECT decisions

MIInfer should copy this methodology.

---

## Vega20-specific MMQ configuration

The project contains dedicated Vega/gfx906 MMQ configuration.

Important observation:

> gfx906 may have occupancy constrained by LDS before VGPR count becomes the limiting factor.

This means reducing VGPRs purely to increase theoretical occupancy may be useless if LDS already permits only one resident workgroup per CU.

### MIInfer implication

Benchmark:

* tile size
* LDS use
* VGPR count
* launch bounds
* actual throughput

Do not optimize occupancy as an isolated metric.

---

## Prefetch experiment

The project tested a gfx906 prefetch optimization borrowed from another fork.

The optimization regressed its workload significantly.

The likely reason was workload mismatch:

* original workload benefited from prefetch
* target workload already had useful cache residency
* added loads/instructions/register pressure became overhead

### MIInfer implication

This is one of the project's most important methodological lessons:

> Even MI50-specific optimizations are not portable between MI50 workloads.

Every optimization must be validated against the exact MIInfer workload.

---

## Hardware-state contamination

The project documented benchmark runs where GPU clocks became stuck at abnormally low states after repeated process failures.

Performance dropped dramatically even though source code was not responsible.

### MIInfer implication

Benchmark automation must capture:

* SCLK
* MCLK/HBM clock
* temperature
* power
* GPU state

Performance results without hardware-state validation may be invalid.

---

## Graph topology

The project observed a large performance regression from a small graph-shape change that appeared semantically unimportant.

Reverting the graph change restored performance.

### MIInfer implication

Graph topology itself is a performance parameter.

Do not dismiss measured regressions because source-level reasoning suggests a change should be harmless.

This strengthens MIInfer's static-plan strategy.

---

## Adaptive speculative/MTP work

The project demonstrates that speculative decoding can provide major gains at short context but substantially smaller gains at deep context.

### MIInfer implication

Speculative decoding belongs later in the roadmap.

First optimize baseline execution.

Context depth must be included in speculative-decoding benchmarks.

---

## What to take

* experimental methodology
* Vega20 MMQ reasoning
* benchmark discipline
* contamination detection
* static configuration mindset
* graph-shape sensitivity
* negative-result preservation
* strong external baseline

## What not to take

* llama.cpp as MIInfer's runtime architecture
* every optimization found in the fork
* multi-GPU assumptions from its benchmark environment

---

# 3. iacopPBK/llama.cpp-gfx906

**Repository:** `iacopPBK/llama.cpp-gfx906`

## Status

Historical gfx906 optimization project.

The project itself indicates that development moved elsewhere and the original fork should not be considered the long-term current implementation.

Nevertheless, it contains substantial gfx906-specific engineering knowledge.

## Role

Research source and optimization-idea repository.

Not recommended as MIInfer's architectural base.

---

## gfx906 primitive layer

The project contains reusable architecture-specific mechanisms around:

* DPP
* lane broadcast
* `ds_swizzle`
* reductions
* native math operations
* packed dot products

### MIInfer implication

Build a small reusable gfx906 primitive layer rather than duplicating low-level ISA logic across kernels.

Potential direction:

```text
gfx906/
└── primitives/
    ├── lane
    ├── dpp
    ├── swizzle
    ├── reduction
    ├── dot
    ├── math
    └── memory
```

---

## Logical half-wave execution

One important technique uses a physical Wave64 while dividing it into multiple logical cooperative groups.

For example:

```text
Wave64

lanes 0..31   → output row A
lanes 32..63  → output row B
```

### MIInfer implication

Benchmark:

* 64-lane cooperation
* 32-lane logical groups
* potentially 16-lane groups

Do not assume physical wave width equals optimal logical work width.

---

## Q4 × Q8 packed dot product

The project provides strong support for the hypothesis that quantized inference on gfx906 may benefit from:

```text
Q4 weight
+
Q8 activation
↓
packed integer dot
↓
INT32 accumulation
↓
scale
```

with `v_dot4_i32_i8` as an important candidate primitive.

### MIInfer implication

This should be one of the first serious M2 experiments.

However, the instruction itself is not the optimization.

The benchmark must include:

* unpacking cost
* scale cost
* register pressure
* memory behavior
* final numerical accuracy

---

## Software prefetch

The project contains explicit prefetch experiments.

Other gfx906 projects later demonstrated that similar prefetch techniques can regress different workloads.

### MIInfer implication

Prefetch remains an experiment, not an architectural assumption.

---

## Quantized activation reuse

The project experimented with caching Q8 activations when several operations consume the same source tensor.

### MIInfer implication

The underlying idea is useful.

The dynamic cache architecture is not.

MIInfer's fixed graph should instead represent this statically:

```text
RMSNorm
   ↓
Q8 activation
   ├── Q
   ├── K
   └── V
```

No runtime graph scanning should be required.

---

## FlashAttention and q8 attention

The project contains specialized attention work for gfx906.

### MIInfer implication

Use it as:

* implementation research
* correctness reference
* source of configuration ideas

Do not start MIInfer by reproducing this work unless profiling shows attention is an important bottleneck.

---

## What to take

* primitive-layer idea
* half-wave execution
* Q4×Q8 hypothesis
* DPP/swizzle techniques
* static gfx906 specialization
* attention implementation knowledge

## What not to take

* full fork architecture
* dynamic activation cache machinery
* every prefetch optimization
* assumptions tied to its exact model workload

---

# 4. mxxm-t/mx-llama.cpp

**Repository:** `mxxm-t/mx-llama.cpp`

## Role

Successor direction referenced by earlier gfx906 optimization work.

Important primarily for:

* continued gfx906 optimization
* weight repacking
* activation reuse
* quantized execution
* model-specific tuning

---

## Weight repacking

Reported work suggests significant benefits can come from changing weight layout rather than merely changing arithmetic instructions.

One example strategy separates or reorganizes quantized values and scale data into layouts that are more efficient for gfx906 kernels.

### MIInfer implication

Weight layout is a first-class optimization problem.

Benchmark:

```text
canonical format
vs
MI50-native packed format
```

before assuming model-storage formats are suitable execution formats.

---

## MXFP4

The project reports useful improvements from MXFP4-related work for some workloads.

### MIInfer implication

Potential later M2/M7 candidate.

Initial quantization research should consider:

* Q4
* Q6
* Q8
* MXFP4 where model compatibility warrants it

Do not add multiple formats before the first core execution path is understood.

---

## Activation reuse

Simplified quantized activation reuse reportedly survives in successor work.

### MIInfer implication

This strengthens the argument for planner-managed activation reuse.

---

## What to take

* weight-layout research
* MXFP4 as future candidate
* simplified activation reuse
* modern gfx906 tuning results

## What not to take

* assumption that reported percentages transfer directly to MI50 target workload
* full llama.cpp architecture

---

# 5. ai-infos/vllm-gfx906-mobydick

**Repository:** `ai-infos/vllm-gfx906-mobydick`

## Role

Modern gfx906 compatibility and failure-map research.

This project is valuable because it attempts to run newer model stacks on hardware no longer prioritized by the mainstream ROCm ecosystem.

It should be treated primarily as:

> a database of gfx906 problems and workarounds.

Not as MIInfer's runtime blueprint.

---

## Toolchain lessons

The project uses a carefully assembled gfx906-compatible software stack rather than assuming current default packages work automatically.

### MIInfer implication

The ROCm/toolchain combination is part of the platform.

Pin it.

Do not assume:

```text
newer ROCm = better gfx906
```

---

## BF16

The project highlights poor/unsupported BF16 behavior on gfx906 in modern software paths.

### MIInfer implication

Do not use BF16 merely because model configuration requests it.

Prefer explicit precision selection.

Initial floating-point path:

```text
FP16 operand
FP32 accumulation where required
```

---

## GPTQ / AWQ

Historical gfx906 vLLM work investigated:

* GPTQ
* AWQ
* INT4
* INT8
* specialized GEMV
* Triton tuning

Results varied heavily by workload.

### MIInfer implication

Quantization format alone does not guarantee fast gfx906 execution.

Kernel architecture and exact shapes matter.

---

## Skinny GEMV/GEMM

Historical work demonstrated substantial importance for specialized skinny matrix execution.

### MIInfer implication

For batch-1 decode, prioritize:

* GEMV
* skinny GEMM

over optimizing giant GEMM workloads that may not represent decode.

---

## Accumulator precision

Some experiments required FP32 accumulation to avoid numerical corruption.

### MIInfer implication

Precision must be represented per operation.

Do not globally optimize for lowest precision.

---

## MoE

The project documents challenges around quantized MoE execution.

### MIInfer implication

If MIInfer's first performance target is a MoE model, MoE requires its own profiling and kernel strategy.

Do not assume dense-model optimizations automatically transfer.

---

## Long-context correctness

Historical work encountered failures that appeared only at longer context.

### MIInfer implication

Correctness testing must extend beyond small synthetic tensors and short generations.

Eventually test:

* longer generations
* long context
* repeated execution

---

## Triton-gfx906

The project demonstrates that a gfx906-compatible Triton environment can be useful for prototyping modern kernels.

### MIInfer implication

Triton may be useful during experimentation:

```text
hypothesis
 ↓
Triton prototype
 ↓
benchmark
 ↓
HIP/ISA implementation
```

but should not automatically become a production dependency.

---

## What to take

* compatibility failure map
* precision lessons
* MoE lessons
* long-context correctness lessons
* Triton prototyping strategy
* exact-shape tuning mindset

## What not to take

* vLLM scheduler architecture
* continuous batching assumptions
* Python-heavy hot paths
* generic serving architecture

---

# 6. nlzy/vllm-gfx906

**Repository:** `nlzy/vllm-gfx906`

## Status

Historical/archive reference.

## Role

Important historical source for gfx906 optimization experiments and model-compatibility findings.

Research from this project feeds into the lessons preserved by later projects such as MobyDick.

---

## Key historical observations

Relevant findings include:

* quantized MoE can be difficult to optimize
* unquantized skinny GEMV can benefit substantially from specialization
* AWQ/GPTQ results vary strongly by workload
* accumulator precision affects correctness
* GGUF quantized GEMV can benefit from dedicated paths
* some Triton tuning can help substantially
* model-specific tuning files matter

### MIInfer implication

There is no single "gfx906 optimization."

Performance depends on:

```text
model
+
tensor shape
+
quantization
+
context
+
kernel
+
memory layout
```

---

# 7. Neroued/ninfer

**Repository:** `Neroued/ninfer`

## Role

Architectural inspiration rather than implementation source.

NInfer demonstrates the value of specializing aggressively for:

* exact GPU architecture
* exact model checkpoints
* fixed runtime capacity
* custom packed model artifacts
* static memory planning
* specialized serving behavior

NInfer targets modern NVIDIA hardware, so its kernels are not directly transferable.

Its philosophy is.

---

## Selected models, selected hardware

NInfer's core idea is effectively:

> Selected checkpoints. Maximum single-GPU inference performance.

### MIInfer implication

This aligns strongly with MIInfer's intended direction.

Do not optimize for arbitrary models before proving one exact target.

---

## Packed artifact

NInfer uses its own packed model artifact.

### MIInfer implication

A future MIInfer artifact could similarly store:

* model metadata
* tokenizer
* MI50-native packed weights
* alignment information
* format version

Potential direction:

```text
HF / GGUF
   ↓
MIInfer converter
   ↓
.mi50
```

This belongs after packing experiments demonstrate value.

---

## Fixed memory planning

NInfer benefits from knowing its execution shape and capacity.

### MIInfer implication

MIInfer should similarly prefer:

```text
allocate once
plan once
execute repeatedly
```

---

## Graph-based decode

NInfer uses CUDA graph mechanisms for repetitive execution.

### MIInfer implication

Evaluate HIP graph capture once MIInfer decode becomes stable.

---

## What to take

* specialization philosophy
* custom packed artifact idea
* fixed model support
* fixed memory planning
* graph replay mindset

## What not to take

* CUDA implementation
* NVIDIA-specific kernels
* architecture decisions tied to modern tensor cores

---

# 8. anikifoss/llama.cpp-gfx906

**Repository:** `anikifoss/llama.cpp-gfx906`

## Role

Important gfx906 performance reference, especially for modern Qwen/MoE workloads.

The project demonstrates that significant practical inference performance remains possible on MI50-class hardware with targeted optimization.

---

## Relevant optimization areas

Reported work includes:

* Wave64 specialization
* register blocking
* packed dot operations
* `DS_SWIZZLE`
* attention specialization
* quantized KV experiments
* Qwen-specific tuning

### MIInfer implication

Useful as:

* benchmark reference
* research source
* target-model comparison

Do not assume its implementation choices are optimal for MIInfer.

---

## Qwen3-30B-A3B

This project has demonstrated practical single-MI50 execution of Qwen3-30B-A3B-class quantized models.

### MIInfer implication

This strengthens the case for using a modern MoE model as an eventual MIInfer performance target.

However, the exact first target should be frozen only after M0 baseline work.

---

## KV precision

Reported performance differs materially depending on KV precision.

### MIInfer implication

KV format is both:

* a memory decision
* a performance decision

It must be benchmarked independently.

---

# 9. thickprogrammer/llama.cpp-gfx906

**Repository:** `thickprogrammer/llama.cpp-gfx906`

## Role

Additional gfx906 optimization reference.

Use primarily for:

* implementation comparison
* kernel ideas
* build/toolchain knowledge
* independent confirmation of gfx906-specific techniques

Do not treat overlapping techniques across forks as proof of superiority.

Independent benchmarking remains required.

---

# 10. nick413-bit/gfx906-fa-vllm

**Repository:** `nick413-bit/gfx906-fa-vllm`

## Role

Research source for gfx906 attention and MoE kernel ideas.

The project highlights the importance of gfx906 instructions such as:

* `v_dot2_f32_f16`
* `v_dot4_i32_i8`

and explores W4A16-style execution strategies.

## MIInfer implication

Useful source of kernel hypotheses.

Projected gains should be treated as hypotheses until reproduced on the MI50 target.

---

# 11. joe2gaan/localaiservers

**Repository:** `joe2gaan/localaiservers`

## Role

Current community work preserving modern inference on gfx906-era hardware.

Especially relevant because it includes direct gfx906 microbenchmark observations.

---

## Important dot-product lesson

A reported dominant decode GEMV was tested with a `v_dot2`-based replacement.

The low-level instruction substitution did **not** automatically improve performance.

### MIInfer implication

This is a central project rule:

> ISA-level cleverness is not performance evidence.

Every instruction-level hypothesis must survive real measurement.

---

# 12. AMD gfx906 ISA Documentation

## Role

Authoritative source for:

* available instructions
* execution behavior
* register semantics
* memory operations
* DPP behavior
* packed dot instructions

Use ISA documentation when implementing low-level primitives.

Do not infer instruction availability from newer AMD architectures.

---

# 13. AMD HIP / ROCm Documentation

## Role

Reference for:

* HIP runtime behavior
* device APIs
* graph capture
* events/timing
* compiler configuration
* memory APIs

Because gfx906 support changes across ROCm versions, documentation for the currently validated toolchain is more relevant than assumptions based solely on the newest release.

---

# 14. rocBLAS / hipBLAS

## Role

Potential:

* correctness baseline
* FP16/GEMM performance baseline
* fallback for non-critical operations during early development

MIInfer should benchmark exact shapes before deciding whether custom kernels are necessary.

Do not assume:

```text
custom = faster
```

or:

```text
rocBLAS = faster
```

Measure both.

---

# 15. Research Classification

External ideas should be classified before implementation.

## Category A — Architecture lessons

Examples:

* static planning
* packed artifacts
* specialization
* fixed memory plans

These influence MIInfer's architecture.

---

## Category B — Kernel hypotheses

Examples:

* Q4×Q8 dp4a
* DPP reductions
* half-wave execution
* weight repacking
* explicit prefetch

These require experiments.

---

## Category C — Failure lessons

Examples:

* BF16 fallback
* long-context corruption
* clock-wedged benchmarks
* graph-shape regressions
* precision-sensitive MoE

These should influence validation and tooling.

---

## Category D — Runtime features

Examples:

* MTP
* speculative decoding
* KV compression
* serving

These belong later unless profiling proves otherwise.

---

# 16. Current Research Synthesis

The current external-project landscape can be summarized as:

```text
NInfer
   ↓
teaches us WHAT to specialize

MobyDick / historical vLLM gfx906 work
   ↓
teaches us WHERE gfx906 breaks

iacopPBK / mx-llama.cpp
   ↓
teaches us HOW Vega20 can be specialized

milpster
   ↓
teaches us HOW TO PROVE an optimization is real

llama.cpp
   ↓
provides the general reference/oracle
```

MIInfer should combine those lessons without inheriting any single project's architecture wholesale.

---

# 17. Highest-Priority Research Ideas

Current high-priority candidates for isolated experiments:

## P0

* FP16 GEMV baseline
* quantized GEMV baseline
* Q4 × Q8 packed dot execution
* Wave64 vs logical half-wave execution
* MI50-native packed weight layout

## P1

* activation quantization reuse
* skinny GEMM
* DPP reduction primitives
* LDS/VGPR configuration studies
* launch-bound configuration
* static shape-specific kernel tables

## P2

Once model profiling exists:

* MoE fused execution
* attention specialization
* Q8 K / FP16 V
* HIP graph replay

## Later

* speculative decoding
* adaptive MTP
* custom packed model artifact
* long-context policy
* second model
* multi-GPU

---

# 18. Research Intake Checklist

Before porting an external optimization, answer:

1. What problem does it solve?
2. Which hardware was it measured on?
3. Which model was used?
4. Which tensor shapes were involved?
5. Which quantization was involved?
6. Was the workload prefill or decode?
7. What context length was used?
8. What was the measured baseline?
9. What profiler evidence supported the change?
10. Does MIInfer currently have the same bottleneck?
11. Can the optimization be isolated?
12. What is its implementation complexity?
13. What correctness risks exist?
14. What VRAM cost exists?
15. What would cause us to reject it?

If these questions cannot be answered, research the idea further before implementing it.

---

# 19. External Code Policy

Studying another implementation is encouraged.

Copying code requires explicit license review.

When external code is adapted:

* preserve required copyright notices
* preserve required license text
* document significant provenance
* do not remove attribution

When only an optimization concept is used, prefer implementing it in MIInfer's own architecture where practical.

---

# 20. Research Notes vs Decisions

This file records external evidence and ideas.

It does not automatically establish architectural decisions.

Major project choices belong in:

```text
docs/decisions.md
```

Performance conclusions belong in:

```text
experiments/
```

Current implementation priorities belong in:

```text
docs/current-state.md
```

---

# 21. Updating This Document

Add a project or paper when it provides one or more of:

* relevant gfx906 implementation
* useful performance evidence
* useful failure evidence
* relevant model compatibility information
* reusable architecture insight

For every new reference, document:

```text
role
useful lessons
ideas worth testing
ideas not worth inheriting
```

Avoid turning this file into an unannotated link collection.

---

# 22. Guiding Rule

The purpose of external research is not to collect optimizations.

It is to reduce the number of wrong experiments MIInfer needs to perform.

External results generate hypotheses.

Only MIInfer measurements on the target hardware turn those hypotheses into project knowledge.
