# MIInfer Current State

This document describes the **current implementation state** of MIInfer.

It is intentionally operational and should be updated whenever the active milestone, immediate target, or project constraints change.

For long-term direction, see:

* [`roadmap.md`](roadmap.md)
* [`architecture.md`](architecture.md)
* [`benchmarking.md`](benchmarking.md)
* [`hardware.md`](hardware.md)

---

# Current Phase

**M0 — Baseline and Project Bootstrap**

MIInfer is not yet an inference runtime.

The current objective is to establish the build, benchmark, correctness, and hardware-observation infrastructure required for credible gfx906 kernel research.

---

# Current Hardware Target

```text id="f0sqdr"
AMD Instinct MI50 32GB
gfx906 / Vega20
Linux
single GPU
```

No other GPU architecture is currently supported.

---

# Current Project Status

## Completed

* project mission defined
* specialization strategy defined
* `AGENTS.md`
* `README.md`
* roadmap defined
* architecture boundaries defined
* benchmarking standard defined
* hardware target defined
* C++20/CMake gfx906 build presets
* host-only and GPU-required CTest infrastructure
* trivial HIP vector-add validation
* HIP-event microbenchmark and JSON output
* raw per-iteration benchmark samples
* environment capture and benchmark runner
* active-run GPU telemetry sampling
* EXP-0001 benchmark harness scaffold
* official Qwen3-8B dense-control revision and exact configuration record
* Qwen3-8B projection-shape record for EXP-0002
* EXP-0002 FP16 GEMV baseline scaffold

## In progress

* physical MI50 execution of the bootstrap smoke test and benchmark
* physical MI50 build/validation of the pinned external gfx906 reference
  baseline
* reproducible Qwen3-8B F16 GGUF conversion and checksum
* initial llama.cpp PP/TG measurements

The current environment has not closed these gates: `/dev/kfd` and render
nodes are present, but `rocminfo` fails HSA initialization with a generic
error, so HIP reports no usable device. The pinned reference also cannot
configure here because the hipBLAS CMake development package is absent. The
16.4 GiB Qwen3-8B source snapshot and resulting F16 GGUF were not completed.

## Not implemented

* production C++ runtime
* production HIP kernel library beyond the vector-add validation kernel
* model loading
* GGUF parsing
* tensor packing
* execution planner
* memory planner
* tokenizer
* sampling
* attention
* MoE execution
* HIP graph capture
* CLI
* HTTP server

The C++20/HIP infrastructure is present, but no model/runtime functionality has
been implemented.

---

# Immediate Objective

The immediate technical objective is:

> Close M0 through physical MI50 validation and establishment of the external
> gfx906 reference baseline.

The repository-side infrastructure is complete. Remaining work is execution
and evidence collection on the target machine, followed by freezing the
reference configuration.

---

# Immediate Deliverables

The current bootstrap phase should produce:

* root CMake project
* canonical gfx906 build preset
* trivial HIP kernel
* device validation
* CPU-side correctness test infrastructure
* GPU test infrastructure
* microbenchmark harness
* machine-readable benchmark output
* hardware/environment capture
* experiment scaffold

The repository-side deliverables are complete. Physical MI50 execution remains
required for the GPU-specific exit criteria.

---

# Current Build Direction

The intended build stack is:

```text id="v2eslv"
CMake
C++20
HIP
gfx906
```

Canonical configurations include:

```text id="1kk4ya"
mi50-debug
mi50-release
```

Do not introduce additional build systems unless explicitly justified.

The release preset explicitly compiles for `gfx906`. A host-only preset is
available for environments without a HIP toolchain; the canonical MI50 presets
require HIP.

---

# Current Dependency Policy

Dependencies should remain minimal.

Do not add:

* llama.cpp
* GGML
* PyTorch
* vLLM
* Triton runtime dependency
* Boost
* large framework libraries

unless explicitly approved for a specific purpose.

Small test or utility dependencies may be considered if they reduce complexity without affecting runtime architecture.

---

# Current Runtime Policy

There is currently **no runtime architecture to implement beyond what is required by M0/M1 infrastructure**.

Do not prematurely create:

* generic graph abstractions
* scheduler frameworks
* backend interfaces
* plugin systems
* model registries
* generic tensor frameworks

The architecture should emerge from measured kernel and model requirements.

---

# Current Reference Strategy

MIInfer will maintain a separate external gfx906 reference implementation for:

* performance comparison
* correctness comparison
* model behavior reference
* research

The primary reference is pinned to `milpster/gfx906-llama-cpp` commit
`6e4ef6c1a553b8f61ad77bba18e9ca05aa677295`. The dense control model is pinned
to `Qwen/Qwen3-8B` revision
`b968826d9c46dd6066d109eabc6255188de91218`. The physical MI50 build,
conversion artifact, and measurements remain pending; see
[`reference-baseline.md`](reference-baseline.md).

The reference implementation is not part of MIInfer's runtime architecture.

---

# Current Benchmark Priority

Initial benchmark work should focus on kernel-level infrastructure.

The first benchmark families should approximately be:

```text id="drjqfz"
1. trivial HIP launch/timing validation
2. memory bandwidth sanity
3. FP16 GEMV baseline
4. quantized GEMV baseline
5. gfx906-specialized GEMV experiments
```

Attention and MoE benchmarks should wait until representative target-model shapes and actual bottlenecks are frozen.

---

# Current Experiment Queue

Provisional experiment sequence:

```text id="a3i4qd"
EXP-0001 — benchmark harness validation

EXP-0002 — FP16 GEMV baseline

EXP-0003 — quantized GEMV baseline

EXP-0004 — gfx906-specific Q4 × Q8 GEMV

EXP-0005 — Wave64 vs logical half-wave execution

EXP-0006 — kernel-native weight packing
```

The exact ordering may change based on early measurements.

---

# First Major Gate

The first major project decision occurs at **M2 — Prove Specialization**.

Before significant runtime implementation begins, MIInfer must demonstrate credible evidence that gfx906-specific specialization can improve important target-model operations.

If M2 fails to show meaningful potential, the project should be reassessed rather than automatically continuing into a full runtime.

---

# Current Correctness Policy

All candidate kernels must be validated against a trusted reference implementation.

Performance measurements from incorrect kernels are invalid.

Initial kernel tests should include:

* deterministic input generation
* CPU/reference output
* GPU output
* tolerance-based comparison
* explicit NaN/Inf detection

---

# Current Performance Policy

Do not accept performance claims from:

* one run
* unverified GPU clocks
* mismatched build flags
* mismatched tensor shapes
* mismatched quantization
* contaminated hardware state

Follow [`benchmarking.md`](benchmarking.md).

---

# Current Hardware Observation Requirements

Before meaningful GPU benchmarks are accepted, the project should be able to capture where available:

* GPU identity
* gfx architecture
* VRAM
* ROCm version
* HIP compiler version
* kernel version
* SCLK
* MCLK/HBM clock
* temperature
* power
* power limit

Unavailable metrics should be reported as unavailable, not guessed.

---

# Current Scope

## In scope now

* C++20
* HIP
* gfx906
* MI50
* benchmark infrastructure
* correctness infrastructure
* hardware-state capture
* low-level kernel experiments

## Not in scope now

* model serving
* OpenAI API compatibility
* speculative decoding
* MTP
* multimodal inference
* multi-GPU
* distributed inference
* generic model support
* Windows
* CUDA
* RDNA
* MI200/MI300
* training
* fine-tuning

---

# Do Not Start Yet

Until the roadmap explicitly advances, do not spend implementation effort on:

```text id="eofjf1"
tokenizer
HTTP server
OpenAI-compatible API
generic GGUF support
multi-model support
multi-GPU
speculative decoding
MTP
continuous batching
distributed scheduling
```

These do not help answer the current project question.

---

# Next Implementation Task

The next Codex task should be:

> Close M0 through physical MI50 validation and reference-baseline
> establishment.

Expected scope:

* run the Debug and Release builds on the physical MI50
* execute host and GPU CTest checks
* execute EXP-0001 with repeated benchmark runs
* inspect active-run telemetry for clock/thermal contamination
* pin `milpster/gfx906-llama-cpp` and record its build configuration
* choose and document the initial baseline model

Do not begin EXP-0002 or implement LLM inference until this evidence is
recorded.

---

# Definition of Current Success

The current phase succeeds when a contributor can:

```text id="f4dg36"
clone MIInfer
     ↓
configure canonical MI50 build
     ↓
compile gfx906 HIP code
     ↓
run correctness tests
     ↓
run a microbenchmark
     ↓
capture hardware state
     ↓
produce reproducible benchmark output
```

Only then should the project begin serious kernel specialization work.

---

# Last Updated

2026-08-29 — repository bootstrap implemented; MI50 execution pending.

Update this document whenever:

* active milestone changes
* immediate technical objective changes
* supported hardware changes
* experiment priority changes
* a major architectural assumption changes
