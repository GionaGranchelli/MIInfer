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

**M2 — Prove Specialization**

MIInfer is not yet an inference runtime.

M0 is closed and M1 established the kernel laboratory. M2 is the active
go/no-go phase for gfx906-specific specialization. MIInfer is not yet an
inference runtime.

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
* EXP-0003 FP16 GEMV bottleneck characterization diagnostics

## Completed in Task 3

* physical MI50 execution with gfx802 isolated from KFD
* physical MI50 Debug and Release GPU validation
* EXP-0001 with five valid runs and active telemetry
* physical MI50 build and validation of the pinned external gfx906 reference
* reproducible Qwen3-8B F16 GGUF conversion and checksum
* initial llama.cpp PP/TG measurements
* EXP-0002 FP16 GEMV baseline implementation and five-run MI50 measurement
* EXP-0004 FP16 K-split K/V specialization, accepted after five-run measurement
* EXP-0005 Q4_0 × Q8_1 quantized GEMV baseline, accepted after five-run measurement
* EXP-0006 packed-dot ISA proof, correctness validation, and five-run comparison

EXP-0002 is accepted as `KEEP`. The seven real Qwen3-8B projection shapes are
correctness-valid for both the project-owned HIP baseline and the strongest
valid installed-library comparison (`hipblasGemmEx` with FP16 operands and
FP32 compute). The canonical streaming results and raw artifacts are recorded
in [`EXP-0002`](../experiments/EXP-0002-fp16-gemv-baseline.md).

ROCr/HSA initialization fails with both AMD GPUs exposed, but the confirmed
gfx802-isolation workaround leaves the MI50/gfx906 device usable. With that
configuration HIP, MIInfer, the pinned reference, and the Qwen3-8B smoke test
all work. The gfx802 isolation remains an operational platform prerequisite.

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

> Use measured evidence to select the next gfx906-specific quantized
> specialization after EXP-0006 rejected the current register-unpack
> packed-dot implementation. EXP-0005 remains the frozen correctness and
> performance baseline; M2 is not yet passed.

M0 is closed under the documented gfx802-isolated configuration. The
repository-side infrastructure, physical MI50 validation, model artifact, and
reference baseline are recorded. The gfx802-isolation requirement remains a
documented platform prerequisite for M1 GPU execution.

---

# Immediate Deliverables

The M1/M2 kernel-laboratory deliverables currently include:

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
* deterministic Q4_0/Q8_1 host quantization and CPU oracle
* project-owned Q4_0 × Q8_1 HIP baseline
* activation-quantization and fan-out measurements
* gfx906 `v_dot4_i32_i8` probe and Q4×Q8 packed-dot candidate
* EXP-0006 five-run scalar-versus-packed-dot evidence

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
conversion artifact, and measurements are recorded in
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

The accepted EXP-0002 baseline has been characterized in EXP-0003, and
EXP-0004 established a K/V-specific FP16 K-split specialization with a 52%
latency reduction on the real `M=1024, K=4096` shapes. The
external llama.cpp kernel-share measurement remains a `RETEST` item because a
compatible ROCm profiler is not installed, but that gap does not block
isolated kernel experiments. It remains required before an M2 or end-to-end
claim against the real llama.cpp decode path.

Attention and MoE benchmarks should wait until representative target-model
shapes and actual bottlenecks are frozen. EXP-0005 accepted the Q4_0 × Q8_1
baseline on all seven real Qwen3-8B projection shapes. EXP-0006 proved that
the compiler emits `v_dot4_i32_i8` and the candidate is numerically correct,
but the current register unpack/pack implementation regresses Q/O and FFN
latency. The accepted K-split implementation is currently limited to the K/V
shape family; Q/O continues
to use the EXP-0002 baseline configuration.

---

# Current Experiment Queue

Provisional experiment sequence:

```text id="a3i4qd"
EXP-0001 — benchmark harness validation

EXP-0002 — FP16 GEMV baseline

EXP-0003 — FP16 GEMV bottleneck characterization (RETEST: external profiler gap)

EXP-0004 — FP16 K-split parallelism for K/V (KEEP)

EXP-0005 — quantized GEMV baseline (KEEP)

EXP-0006 — gfx906 Q4_0 × Q8_1 packed-dot specialization (REJECT)

EXP-0007 — reduce Q4 register-unpack/pack overhead (recommended)
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

> Decide whether to proceed to M1 kernel-laboratory work or first pursue a
> durable ROCr fix that permits the gfx802 display GPU to remain attached.

The M0 evidence gates are complete under the documented gfx802-isolated
configuration. Do not begin EXP-0002 or implement LLM inference until the
project explicitly advances to M1.

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

2026-08-29 — M0 closed with gfx802 isolated from KFD; M1 ready.

Update this document whenever:

* active milestone changes
* immediate technical objective changes
* supported hardware changes
* experiment priority changes
* a major architectural assumption changes
