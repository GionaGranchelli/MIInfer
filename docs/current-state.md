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

**M4 — First Correct Qwen3-8B Generation**

MIInfer is not yet an inference runtime.

M0 is closed, M1 established the kernel laboratory, M2 passed its
gfx906-specific specialization gate with EXP-0009, and M3 is closed. The
minimal model/runtime scaffold is present; M4 now owns first token generation.

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
* EXP-0007 zero-point-corrected Q4_0 × Q8_1 dot4 specialization, accepted after five-run measurement
* EXP-0008 direct comparison with the pinned gfx906 llama.cpp MMVQ path; the
  reference primitive is now measured
* EXP-0009 128-thread K/V geometry and Wave64 reduction comparison; accepted
  with `KEEP`, with M2 marked `GO`
* M3 minimal Qwen3-8B GGUF loader, GPU weight arena, and static plan; closed
  after pinned physical-MI50 acceptance
* M4-A5 independent four-position reference trace; host and MI50 Debug/Release
  stateful layer-0 comparisons pass
* M4-B initial full-depth host/GPU executor scaffold and independent
  single-token 36-layer reference fixture; acceptance remains open
* M4-B6 independent same-run external layer-6 trace and first-divergence
  host/MI50 diagnostic comparison; full-depth acceptance remains open
* M4-B8 canonical full-depth external oracle and exact-Q8 F16/F32 precision
  policy diagnostics
* M4-B9 terminal layer-35 external internal trace and first-divergence
  comparison; shared host/GPU FFN-tail numeric contract remains open;
  production precision remains unchanged
* M4-B10 host Gate/Up hybrid SwiGLU attribution; both projection errors are
  causal, with Gate the larger single-source contributor, while host SwiGLU
  arithmetic is exonerated
* M4-B11 Q8 contract and Gate/Up accumulation replay; Q8 lanes/scales and
  external-conditioned Gate/Up projection arithmetic match the pinned
  contract, shifting the remaining failure upstream to `ffn_norm`
* M4-B12 external-conditioned O/residual and RMSNorm replay; O/residual and
  all tested RMSNorm reductions are close to or exact with external inputs,
  shifting the remaining normal-path difference upstream to attention output
* M4-B13 attention RMSNorm/V/GQA replay; V and GQA are exonerated, while the
  external attention-output FP16 materialization reproduces the remaining
  layer-35 tail result

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

* full production inference runtime and model-facing integration
* model loading outside the pinned Qwen3-8B contract
* general GGUF parsing
* custom tensor packing
* production execution planner
* general-purpose memory planner
* tokenizer
* accepted full multi-layer model execution
* token generation
* sampling
* MoE execution
* HIP graph capture
* CLI
* HTTP server

The C++20/HIP infrastructure, model loader/planner, layer-0 correctness
runtime, and an unaccepted full-depth execution scaffold are present. Full
model correctness and token generation remain outside the accepted scope.

---

# Immediate Objective

The immediate technical objective is:

> Close M4-B by resolving the first depth-localized numerical divergence
> between the 36-layer host/MI50 executors and the independent pinned
> reference, then passing the full layer-output, final-norm, and logits gates.
> Do not broaden support beyond the pinned model contract.

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
* zero-point-corrected Q4×Q8 dot4 kernel using Q8_1 sum metadata
* EXP-0007 five-run scalar/control/candidate evidence and size-matched memory reference
* EXP-0008 five-run direct primitive comparison against pinned llama.cpp MMVQ
* EXP-0009 five-run 128-thread and Wave64 geometry comparison against MMVQ
* M3 pinned Qwen3-8B model recognition, 399-tensor validation, 4.77 GB GPU
  weight residency, and static kernel/buffer plan
* M4-A correctness foundation for Qwen3 host oracles and initial gfx906 GPU
  probes (RMSNorm, Q4_0 embedding lookup, and Q6_K GEMV)
* M4-A2 complete host-side layer-0 composition for the pinned single-token
  fixture, automatic comparison of all 28 reference checkpoints, and a
  comparator mutation test
* M4-A3 complete MI50 GPU layer-0 composition for the same position-zero
  fixture, with GPU-to-host and GPU-to-reference comparison in Debug and
  Release, plus a GPU composition mutation discriminator
* M4-A4 deterministic four-position host and MI50 layer-0 execution with an
  explicit post-RoPE KV-cache contract, reset/append/preservation checks,
  causal-prefix validation, and cache mutation discriminators
* M4-A5 independent external four-position reference trace, including
  post-RoPE K/V cache-write vectors, with Host/MI50 Debug/Release parity

The repository-side specialization and M3 runtime-scaffold deliverables are
complete. M4-A is complete: the host and MI50 layer-0 compositions match an
independent four-position trace from the pinned reference within the frozen
stage-specific tolerances in Debug and Release. The external trace also
directly pins post-RoPE K and unmodified V as the cache-write representation.
The four-position tests prove append, preservation, reset, causal extent,
host/GPU cache parity, and external-trace comparator mutation detection. No
token-generation or end-to-end performance claim is made here.

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

There is currently **no token-executing runtime**. M4 may implement only the
minimum execution path required for the selected Qwen3-8B target.

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
external llama.cpp kernel-share profiling remains unavailable because a
compatible ROCm profiler is not installed, but EXP-0008 completed the more
important direct primitive timing for the pinned MMVQ path. At that stage it
did not yet show a material MIInfer advantage across the major Q/O and FFN
regimes; EXP-0009 subsequently corrected the measured geometry gap.

The accepted EXP-0009 result produced a shape-specialized MIInfer family
competitive with or faster than the pinned MMVQ path on all seven projection
shapes.

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

EXP-0007 — zero-point-corrected Q4_0 × Q8_1 dot4 (KEEP)

EXP-0008 — direct MIInfer versus pinned gfx906 llama.cpp MMVQ (KEEP)

EXP-0009 — K/V workgroup and Wave64 reduction geometry (KEEP)

M3 — minimal Qwen3-8B runtime scaffold (CLOSED)

M4 — first correct Qwen3-8B token generation (next milestone)
```

The exact ordering may change based on early measurements.

---

# First Major Gate

The first major project decision occurs at **M2 — Prove Specialization**.

Before significant runtime implementation begins, MIInfer must demonstrate credible evidence that gfx906-specific specialization can improve important target-model operations against the strongest relevant gfx906 implementation. A complete MIInfer runtime is not required for the M2 gate.

If M2 fails to show meaningful potential, the project should be reassessed rather than automatically continuing into a full runtime. EXP-0009 passed this gate with `M2 GO`.

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

The current Codex task is:

> M4-B11 has shown that the pinned Q8 contract and Gate/Up accumulation are
> not the source: with external `ffn_norm`, Gate/Up replay matches external
> outputs to `7.6e-6`/`1.5e-5`. Locate the earlier FFN-input/attention-output
> numeric contract before changing production precision.
> Do not widen tolerances or start token generation.

The M0 evidence gates are complete under the documented gfx802-isolated
configuration. The M2 gate is satisfied by EXP-0009 and M3 is closed by the
pinned real-model acceptance. M4-A is closed, while M4-B full-depth numeric
parity remains open. A full-forward scaffold exists, but no accepted token
generation path has been implemented yet.

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

The M2 validation chain and M3 model-plan acceptance are complete. M4 success
requires correct deterministic token generation without broadening the project
into a generic runtime.

---

# Last Updated

2026-08-31 — M4-B13 attention RMSNorm/V/GQA replay completed; V and GQA are
exonerated and the external attention-output FP16 boundary is the remaining
layer-35 precision hypothesis while full-reference parity remains open.

Update this document whenever:

* active milestone changes
* immediate technical objective changes
* supported hardware changes
* experiment priority changes
* a major architectural assumption changes
