# MIInfer

**MIInfer** is an experimental, performance-first LLM inference runtime designed specifically for **AMD gfx906**, initially targeting the **AMD Instinct MI50 32GB**.

The project explores a simple question:

> How much inference performance can be extracted from gfx906 when the runtime, memory layout, execution plan, and GPU kernels are designed specifically for the architecture instead of treating it as a legacy backend of a general-purpose inference framework?

MIInfer is intentionally specialized.

It is **not** intended to become another general-purpose llama.cpp, vLLM, PyTorch, or generic ROCm inference runtime.

---

## Status

**Current phase: M5 CLOSED — architectural decision gate**

The project currently has:

* a reproducible MI50/gfx906 development environment
* trusted correctness references
* reproducible performance benchmarks, including the first end-to-end M5-A baseline
* kernel-level profiling infrastructure
* a strong llama.cpp/gfx906 comparison baseline
* a pinned Qwen3-8B MI50 execution path with persistent KV-cache decode
* a model-backed tokenizer and minimal text-facing greedy CLI

Sampling and serving remain out of scope. M5 is closed as a measured local
optimization campaign. The next phase depends on the architectural decision
recorded in `EXP-0041`: preserve the current execution contract, or begin
reference-correct execution-contract exploration in M6-A.

The current production path is approximately 55 tok/s at stable peak. No new
kernel experiment is selected until the Path A/Path B decision is made.

---

## Initial target

MIInfer deliberately begins with a narrow target.

| Area             | Initial target                   |
| ---------------- | -------------------------------- |
| GPU              | AMD Instinct MI50 32GB           |
| Architecture     | Vega 20 / gfx906                 |
| Execution        | Single GPU                       |
| Platform         | Linux                            |
| Workload         | LLM inference                    |
| Batch            | Batch 1 / low batch              |
| Models           | Explicitly selected model family |
| Quantization     | Explicitly selected format       |
| Primary language | C++20                            |
| GPU programming  | HIP                              |
| Portability      | Not an initial goal              |

Support for other GPUs, operating systems, model architectures, or execution modes must be justified by project goals and benchmark evidence.

---

## Why MIInfer?

Modern inference frameworks need to support:

* many GPU generations
* many model architectures
* many quantization formats
* dynamic execution graphs
* different batch sizes
* different serving workloads
* multiple accelerator vendors
* distributed execution

Those capabilities are valuable, but they also impose architectural constraints.

gfx906 is now an older and poorly supported architecture in modern ROCm software, yet hardware such as the MI50 still provides:

* 32 GB HBM2
* approximately 1 TB/s memory bandwidth
* Wave64 execution
* useful packed integer instructions
* substantial FP16/FP32 compute capability

MIInfer investigates what becomes possible when we remove most generic-runtime requirements and optimize for the hardware directly.

---

## Core hypothesis

MIInfer is an engineering experiment built around two competing hypotheses.

### H0

A purpose-built gfx906 runtime cannot materially outperform an already well-optimized llama.cpp/gfx906 implementation.

### H1

Removing general-purpose runtime constraints enables meaningful inference improvements on MI50/gfx906.

The project must remain structured so this hypothesis can be tested objectively.

If specialization does not produce meaningful benefits, that is a valid project result.

---

## Design principles

### Specialization is intentional

MIInfer may rely on known properties of:

* gfx906
* Wave64
* MI50 memory characteristics
* supported model dimensions
* supported quantization formats
* fixed tensor layouts

Genericity is not automatically desirable.

---

### Measure before optimizing

No optimization is accepted because it:

* looks theoretically faster
* worked on another GPU
* worked in another gfx906 project
* uses a lower-level ISA instruction
* reduces instruction count
* increases theoretical occupancy

Every meaningful optimization must be measured against a reproducible baseline.

---

### Correctness before performance

A faster kernel that changes model behavior incorrectly is a regression.

Kernel work is validated progressively through:

1. numerical reference comparisons
2. tensor or logits comparisons where practical
3. short generation tests
4. long generation and long-context tests

---

### Negative results are retained

Failed performance experiments are part of the project knowledge base.

An optimization that loses performance should normally be documented rather than silently discarded.

This helps prevent repeated work and makes architectural decisions evidence-based.

---

### Hot paths should be static

Where possible, decisions should happen during model loading rather than during every generated token.

The intended direction is:

```text
model
  ↓
validate supported configuration
  ↓
select kernels
  ↓
construct static execution plan
  ↓
allocate memory
  ↓
load / repack weights
  ↓
execute
```

rather than:

```text
model
  ↓
generic execution graph
  ↓
dynamic operator selection
  ↓
generic scheduler
  ↓
generic accelerator backend
```

---

## What MIInfer intends to own

The project intends to directly control performance-critical architecture such as:

* gfx906 GPU primitives
* quantized matrix/vector kernels
* matrix multiplication kernels
* model-specific kernel selection
* tensor packing and memory layout
* activation reuse
* GPU buffer lifetime planning
* static execution planning
* attention implementation where justified
* MoE execution where relevant
* HIP graph capture strategy
* benchmark methodology

Commodity infrastructure may be reused when doing so does not compromise the experiment.

---

## What MIInfer does not initially aim to build

Initial non-goals include:

* NVIDIA CUDA support
* Intel GPU support
* generic AMD GPU support
* RDNA support
* MI200 / MI300 support
* CPU inference optimization
* Windows support
* macOS support
* training
* fine-tuning
* distributed inference
* tensor parallelism
* pipeline parallelism
* high-concurrency serving
* arbitrary model support
* arbitrary quantization support
* OpenAI-compatible serving
* multimodal inference

These may be reconsidered only after the initial gfx906 hypothesis has been evaluated.

---

## Architecture direction

The planned architecture is deliberately small.

```text
                    MIInfer

                     Model
                       │
                       ▼
                 Model Loader
                       │
                       ▼
              Supported-Model Check
                       │
                       ▼
             Static Execution Planner
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
     Memory Plan               Kernel Plan
          │                         │
          └────────────┬────────────┘
                       ▼
                 gfx906 Runtime
                       │
                       ▼
               gfx906 Kernels
                       │
                       ▼
                AMD Instinct MI50
```

MIInfer does not intend to adopt GGML's generic execution graph or scheduler as its runtime architecture.

Existing frameworks remain important reference implementations and benchmark competitors.

---

## Research influences

MIInfer is informed by several existing projects.

### llama.cpp

Used as:

* correctness reference
* model-format reference
* benchmark baseline
* source of implementation knowledge

MIInfer is not intended to become a llama.cpp fork.

### gfx906 llama.cpp forks

Projects specializing llama.cpp for Vega20/gfx906 provide valuable evidence around:

* Wave64-specific execution
* DPP and swizzle operations
* quantized GEMV
* weight repacking
* attention
* MoE
* HIP graph behavior
* architecture-specific tuning
* failed optimization paths

Ideas from these projects must still be independently benchmarked against MIInfer workloads.

### NInfer

Provides an important architectural lesson:

> Selected hardware and selected models allow much deeper specialization than generic inference frameworks.

MIInfer applies a similar philosophy to gfx906 rather than NVIDIA hardware.

### gfx906 vLLM work

Provides useful information about:

* modern-model compatibility on gfx906
* numerical precision problems
* Triton experimentation
* attention behavior
* MoE bottlenecks
* unsupported ROCm paths

See [`docs/references.md`](docs/references.md) for the research inventory.

---

## Development roadmap

### M0 — Baseline

Establish:

* reproducible MI50 environment
* strongest practical gfx906 reference implementation
* target model
* benchmark methodology
* hardware-state capture
* correctness reference

### M1 — Kernel laboratory

Build:

* HIP benchmark infrastructure
* CPU reference implementations
* numerical validation
* initial GEMV/GEMM experiments

### M2 — Prove specialization

Investigate:

* Q4/Q8 and related packed dot products
* Wave64 and half-wave execution
* DPP/swizzle operations
* architecture-specific memory layouts
* register/LDS trade-offs
* static kernel configuration

This is the first major **go/no-go milestone**.

### M3 — Minimal runtime

Only after sufficient kernel evidence:

* model metadata
* tensor loading
* GPU memory planning
* supported-model validation
* static execution planning

### M4 — First correct generation

Execute one supported model end-to-end.

### M5 — Beat the reference

Compare MIInfer against the strongest reproducible gfx906 baseline.

Primary metrics include:

* prompt processing throughput
* token generation throughput
* time to first token
* VRAM
* context capacity
* power
* tokens per joule

### M6 — Runtime specialization

Potential work:

* native packed model artifacts
* activation reuse
* operation fusion
* fixed buffer reuse
* HIP graph capture
* static decode replay

### M7 — Expansion

Only after the original hypothesis is demonstrated:

* additional quantization formats
* second model
* deeper long-context optimization
* speculative decoding / MTP
* optional serving layer

See [`docs/roadmap.md`](docs/roadmap.md) for the current roadmap.

---

## Benchmark philosophy

Performance measurements are treated as engineering evidence.

Important comparisons should include:

* exact baseline commit
* exact candidate commit
* GPU state
* ROCm version
* compiler version
* model and quantization
* context length
* workload shape
* repeated runs
* correctness verification

For small performance differences, interleaved testing is preferred:

```text
A
B
A
B
A
B
```

rather than running all baseline tests followed by all candidate tests.

Hardware clocks, temperature, and power state must be considered part of benchmark validity.

See [`docs/benchmarking.md`](docs/benchmarking.md).

---

## Experiments

Performance investigations are recorded under:

```text
experiments/
```

Each significant experiment receives an identifier:

```text
EXP-0001
EXP-0002
EXP-0003
...
```

Experiments should document:

* hypothesis
* bottleneck
* baseline
* candidate
* environment
* correctness
* raw measurements
* aggregated measurements
* profiling evidence
* interpretation
* KEEP / REJECT / RETEST decision

Negative experiments are intentionally retained.

---

## Building

The build system is CMake.

The intended canonical development presets are:

```bash
cmake --preset mi50-debug
cmake --build --preset mi50-debug
```

and:

```bash
cmake --preset mi50-release
cmake --build --preset mi50-release
```

Tests:

```bash
ctest --preset mi50-debug
```

The initial benchmark validates HIP execution and timing infrastructure:

```bash
./build/mi50-release/miinfer-bench --warmup 5 --iterations 100
scripts/run-bench.sh ./build/mi50-release/miinfer-bench \
  --warmup 5 --iterations 100
```

The benchmark emits JSON and uses HIP events. The runner stores before/after
environment captures, active-run `rocm-smi` telemetry, and the JSON result under
`bench/results/<run-id>/`.

> The project is currently being bootstrapped. The initial benchmark validates
> infrastructure rather than MIInfer inference performance.

---

## Hardware requirements

The initial supported target is:

```text
AMD Instinct MI50 32GB
gfx906
Linux
ROCm/HIP toolchain capable of producing gfx906 code
```

Current ROCm releases may not provide complete official gfx906 support.

The exact validated development stack will be documented in:

[`docs/hardware.md`](docs/hardware.md)

Do not assume that every modern ROCm library or prebuilt binary contains working gfx906 support.

---

## Repository structure

The intended repository layout is:

```text
MIInfer/
├── AGENTS.md
├── README.md
├── CMakeLists.txt
├── CMakePresets.json
│
├── include/
│   └── miinfer/
│
├── src/
│
├── gfx906/
│   ├── primitives/
│   └── kernels/
│
├── bench/
├── tests/
├── experiments/
├── scripts/
│
└── docs/
    ├── architecture.md
    ├── benchmarking.md
    ├── current-state.md
    ├── decisions.md
    ├── hardware.md
    ├── references.md
    └── roadmap.md
```

The structure will evolve only when implementation evidence requires it.

---

## Contributing

MIInfer is currently in an early research phase.

Before modifying performance-critical code, read:

* [`AGENTS.md`](AGENTS.md)
* [`docs/architecture.md`](docs/architecture.md)
* [`docs/benchmarking.md`](docs/benchmarking.md)
* [`docs/current-state.md`](docs/current-state.md)
* [`docs/decisions.md`](docs/decisions.md)

Performance changes should remain:

* focused
* measurable
* reproducible
* correctness-verified
* easy to compare against a baseline

Do not broaden supported hardware or model scope without an explicit project decision.

---

## License

MIInfer is licensed under the MIT License.

See [`LICENSE`](LICENSE).

External code incorporated into the project must retain all attribution and licensing required by its original license.

---

## Guiding principle

MIInfer favors:

> **simple, explicit, architecture-aware, measured code**

over unnecessary generality.

But specialization without measurement is not optimization.

The project follows this loop:

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

The goal is not merely to make gfx906 run modern LLMs.

The goal is to determine **how fast gfx906 can run them when the software is designed around the hardware**.
