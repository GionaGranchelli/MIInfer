# MIInfer

**MIInfer** is an experimental, performance-first LLM inference runtime designed specifically for **AMD gfx906**, initially targeting the **AMD Instinct MI50 32GB**.

The project explores a simple question:

> How much inference performance can be extracted from gfx906 when the runtime, memory layout, execution plan, and GPU kernels are designed specifically for the architecture instead of treating it as a legacy backend of a general-purpose inference framework?

MIInfer is intentionally specialized.

It is **not** intended to become another general-purpose llama.cpp, vLLM, PyTorch, or generic ROCm inference runtime.

---

## Status

**Current phase: M6-B28 — post-B27 production profile**

The project currently has:

* a reproducible MI50/gfx906 development environment
* trusted correctness references
* reproducible performance benchmarks, including the first end-to-end M5-A baseline
* kernel-level profiling infrastructure
* a strong llama.cpp/gfx906 comparison baseline
* a pinned Qwen3-8B MI50 execution path with persistent KV-cache decode
* a model-backed tokenizer and minimal text-facing greedy CLI
* a completed Qwen3.8-27B GGUF/architecture audit; see
  `experiments/EXP-0042-m6a0-qwen38-27b-gguf-audit.md`
* native Qwen3.8-27B GPU generation through 128 tokens with deterministic
  replay and zero decode-loop allocations; see
  `experiments/EXP-0092-m6a28-native-qwen35-generation.md`
* the first MI50 native-generation throughput baseline and fresh pinned
  llama.cpp comparison; see
  `experiments/EXP-0093-m6b1-qwen35-native-generation-baseline.md`
* a production-selected Q5_K scale/min unpack-hoisting optimization improving
  native TG64/TG128 by about 27%; see
  `experiments/EXP-0095-m6b2-q5k-scale-hoist.md`
* a production-selected Q4_K metadata-staging optimization improving native
  TG64/TG128 by about 38%; see
  `experiments/EXP-0096-m6b3-q4k-metadata-staging.md`
* a production-selected subgroup-structured Q5_K dot loop improving the
  current native TG64/TG128 path by about 26%; see
  `experiments/EXP-0099-m6b6-q5k-subgroup-layout.md`
* a production-selected paired-nibble Q5_K decoding improvement adding about
  2.7% on TG64/TG128; see
  `experiments/EXP-0100-m6b7-q5k-paired-nibbles.md`
* rejected Wave64-local cached-attention reduction and Q6_K LM-head
  index-hoisting candidates, retained as negative evidence; see
  `experiments/EXP-0101-m6b8-attention-wave-local-reduction.md` and
  `experiments/EXP-0102-m6b9-q6k-lm-head-index-hoist.md`
* a rejected recurrent Q8_K input-reuse candidate; see
  `experiments/EXP-0103-m6b10-recurrent-q8-reuse.md`
* a production-selected Q4_K packed-dot4 projection path improving native
  TG64/TG128 by about 9.6%/10.0%; see
  `experiments/EXP-0104-m6b11-q4k-packed-dot4.md`
* a rejected Q6_K packed-dot4 projection experiment: functionally clean but
  end-to-end neutral; see
  `experiments/EXP-0105-m6b12-q6k-packed-dot4.md`
* a rejected Q6_K×Q8_1 LM-head compatibility path: matching representation but
  11.28% slower end-to-end; see
  `experiments/EXP-0106-m6b13-lm-head-q8-1.md`
* a production-selected MMVQ-style Q6_K×Q8_1 LM-head path improving P64 by
  about 3.3%; the former Q6_K×Q8_K path remains available with
  `MIINFER_LM_Q8_1_MMVQ=0` for control comparisons; see
  `experiments/EXP-0107-m6b14-q6k-mmvq-q8-1.md`
* a production-selected recurrent state-update path that avoids the
  intermediate decayed-state store and improves TG64/TG128 by about 2.8%/2.6%;
  the former kernel remains available with
  `MIINFER_DELTA_NO_DECAY_STORE=0`; see
  `experiments/EXP-0108-m6b15-recurrent-no-decay-store.md`
* a production-selected Q8_K projection-input reuse path that quantizes shared
  normalized inputs once for repeated consumers, improving TG64/TG128 by about
  0.8%/0.7%; set `MIINFER_REUSE_PROJECTION_Q8=0` for the separate-quantization
  control; see `experiments/EXP-0109-m6b16-projection-input-q8-reuse.md`
* a production-selected gfx906 Q5_K×Q8_1 MMVQ-style recurrent output
  projection, improving native TG64/TG128 by about 10.9%/10.2%; the former
  Q5_K×Q8_K path remains available with `MIINFER_Q5K_Q8_1_MMVQ=0`; see
  `experiments/EXP-0110-m6b17-q5k-mmvq-q8-1.md`
* a production-selected gfx906 Q4_K×Q8_1 MMVQ-style FFN Down projection,
  improving native TG64/TG128 by about 2.3%/2.2%; the former Q4_K×Q8_K
  path remains available with `MIINFER_Q4K_Q8_1_MMVQ=0`; see
  `experiments/EXP-0111-m6b18-q4k-mmvq-q8-1-ffn-down.md`
* a production-selected shared Q4_K×Q8_1 MMVQ-style FFN Gate/Up path,
  improving native TG64/TG128 by about 5.9%/6.2%; the former Q4_K×Q8_K
  path remains available with `MIINFER_Q4K_Q8_1_MMVQ_FFN_GATE_UP=0`; see
  `experiments/EXP-0112-m6b19-q4k-mmvq-ffn-gate-up.md`
* a rejected Q6_K×Q8_1 MMVQ recurrent QKV candidate: native replay and the
  external observable contract failed immediately, so the B19 Q6_K×Q8_K QKV
  path remains active; see
  `experiments/EXP-0113-m6b20-q6k-mmvq-qkv.md`
* a production-selected Q4_K×Q8_1 MMVQ recurrent gate projection, improving
  native TG64/TG128 by about 1.8%/1.6%; the former Q4_K×Q8_K path remains
  available with `MIINFER_Q4K_Q8_1_MMVQ_ATTN_GATE=0`; see
  `experiments/EXP-0114-m6b21-q4k-mmvq-attn-gate.md`
* a production-selected packed-dot4 Q6_K×Q8_K recurrent QKV projection,
  improving native TG64/TG128 by about 2.9%/2.9%; the scalar path remains
  available with `MIINFER_Q6K_Q8K_DOT4_QKV=0`; see
  `experiments/EXP-0115-m6b22-q6k-q8k-dot4-qkv.md`
* a measurement-only full-attention stage attribution at P64; layer 3 is
  dominated by the combined Q/K/V preparation bucket, FFN, and cached
  attention, with no production behavior change; see
  `experiments/EXP-0116-m6b24-full-attention-attribution.md`
* a measurement-only fine attribution that separates full-attention Q
  projection (~0.199 ms), Q head normalization (~0.085 ms), K/V projections,
  RoPE/KV store, and cached attention; see
  `experiments/EXP-0117-m6b25-full-attention-fine-attribution.md`
* a rejected Q4_K×Q8_1 MMVQ full-attention Q-projection candidate: it failed
  the external observable contract immediately, so the Q4_K×Q8_K path remains
  active; see
  `experiments/EXP-0118-m6b26-q-projection-q8-1-reject.md`
* a production-selected batched full-attention head RMS-normalization path,
  improving stable-peak TG64/TG128 by about 1.4%/1.4%; set
  `MIINFER_BATCH_HEAD_RMS=0` for the separate-launch control; see
  `experiments/EXP-0119-m6b27-batched-head-rms.md`
* a measurement-only post-B27 profile showing 48 recurrent layers consume
  64.54 ms of the instrumented 88.99 ms token and remain the next measured
  target; see `experiments/EXP-0120-m6b28-post-b27-profile.md`
* a reproducible llama.cpp-backed Qwen3.8-27B hybrid tensor/state fixture; see
  `experiments/EXP-0043-m6a1-qwen38-reference-fixture.md`
* a read-only Qwen3.8 projection/kernel compatibility map; see
  `experiments/EXP-0044-m6a2-qwen38-projection-compatibility.md`
* an explicit M6-B1 readiness audit showing that the current qwen3 HIP path
  cannot profile qwen35; see
  `experiments/EXP-0051-m6b1-qwen38-miinfer-gpu-readiness.md`

* L29 recurrent/gated provenance diagnostics through A26.8; see
  `experiments/EXP-0076-m6a266-qwen35-l29-output-provenance.md`,
  `experiments/EXP-0077-m6a267-qwen35-l29-gated-output-provenance.md`, and
  `experiments/EXP-0078-m6a268-qwen35-l29-gate-input-provenance.md`

* a qwen35 model boundary and real MI50 RMSNorm fixture; see
  `experiments/EXP-0052-m6a8-qwen35-gpu-foundation.md`
* a qwen35 Q6_K×Q8_K LM-head projection validated against external logits; see
  `experiments/EXP-0053-m6a9-qwen35-lm-head.md`
* a qwen35 Q4_K×Q8_K attention projection validated against an external
  checkpoint; see `experiments/EXP-0054-m6a10-qwen35-q4k-projection.md`
* a composed qwen35 layer-3 RMSNorm→Q8_K→Q4_K×Q8_K attention prefix; see
  `experiments/EXP-0055-m6a11-qwen35-attention-prefix.md`
* qwen35 layer-3 Q/K/V projections with K normalization validated on MI50; see
  `experiments/EXP-0056-m6a12-qwen35-attention-projections.md`
* a complete qwen35 layer-3 full-attention path through FFN and residual,
  checked at positions 0–8; see
  `experiments/EXP-0057-m6a13-qwen35-full-attention-layer.md`
* qwen35 state fingerprints, poisoned reset, and replay checks for the
  recurrent layers and layer-3 KV cache; see
  `experiments/EXP-0058-m6a14-qwen35-state-audit.md`
* qwen35 layers 0–3 stateful hybrid-block composition through the complete
  layer-3 output boundary at positions 0–16; see
  `experiments/EXP-0059-m6a15-qwen35-hybrid-block-audit.md`
* qwen35 layers 4–7 independently composed after layers 0–3 through position
  16 with recurrent/KV fingerprints; see
  `experiments/EXP-0060-m6a16-qwen35-hybrid-block-4-7-audit.md`
* qwen35 composition ladder through 8, 16, 32, and 64 layers with linear
  one-token scaling and final-logit validation; see
  `experiments/EXP-0061-m6a17-qwen35-composition-ladder.md`
* a real MI50 qwen35 DeltaNet GPU state-update core with persistent
  `[48,128,128]` state validated across positions 0→1; see
  `experiments/EXP-0062-m6a18-qwen35-deltanet-state-gpu.md`
* a real gfx906 qwen35 four-tap convolution, SiLU, Q/K/V split, and Q/K
  normalization path with persistent circular history; see
  `experiments/EXP-0063-m6a19-qwen35-conv-gpu.md`
* a complete layer-0 qwen35 recurrent GPU path through FFN and residual,
  including recurrent projections and beta/alpha preparation; see
  `experiments/EXP-0064-m6a20-qwen35-recurrent-layer-gpu.md`
* qwen35 GPU layers 0–3 composed through the complete full-attention layer-3
  boundary at positions 0→1; see
  `experiments/EXP-0065-m6a21-qwen35-gpu-hybrid-block.md`
* qwen35 GPU layers 0–3 audited through positions 0, 1, 2, 4, 8, 16, 32,
  and 64 with bounded external-reference error and state fingerprints; see
  `experiments/EXP-0066-m6a22-qwen35-gpu-hybrid-position-audit.md`
* qwen35 GPU layers 4–7 composed from the actual L0–L3 output through P64
  using independent recurrent/KV state; see
  `experiments/EXP-0067-m6a23-qwen35-gpu-hybrid-block-4-7.md`
* qwen35 GPU layers 0–7 composed through one common stateful executor through
  P64 with poisoned reset/replay and cached-attention determinism coverage;
  see `experiments/EXP-0068-m6a24-qwen35-eight-layer-gpu-prefix.md`
* qwen35 GPU layers 0–15 composed through the same executor through P64 with
  later recurrent/KV state validation; see
  `experiments/EXP-0069-m6a25-qwen35-sixteen-layer-gpu-prefix.md`
* qwen35 GPU layers 0–31 composed through P64; all layer outputs pass, while
  L30 recurrent-state checkpoints remain in retest; see
  `experiments/EXP-0070-m6a26-qwen35-thirty-two-layer-gpu-prefix.md` and
  `experiments/EXP-0071-m6a261-qwen35-l30-state-localization.md`
* L30 recurrent-update provenance localizes the tracked P63→P64 discrepancy
  to the external-versus-GPU recurrence result; a full P1–P64 scan also finds
  a larger P20 outlier; see
  `experiments/EXP-0072-m6a262-qwen35-l30-update-provenance.md`
* external-input replay clears the MIInfer L30 recurrence itself at the
  P19→P20 worst-case transition; A26 still awaits an explicit state contract;
  see `experiments/EXP-0073-m6a263-qwen35-recurrent-contract-adjudication.md`
* production operand substitution identifies L30 `k_in` as the dominant
  upstream discrepancy at P19→P20; see
  `experiments/EXP-0074-m6a264-qwen35-l30-production-operand-attribution.md`
* L30 K-path provenance finds the first mismatch at its input (`l_out-29`),
  while the K path remains bounded through convolution/SiLU and normalization;
  see `experiments/EXP-0075-m6a265-qwen35-l30-k-path-provenance.md`

Sampling and serving remain out of scope. M5 is closed as a measured local
optimization campaign. M6-A bring-up is complete for the native qwen35 GPU
generation path. M6-B1 now has a first performance baseline; the current
benchmark harness is functional but not yet workload-equivalent enough for a
final parity claim.

The previous Qwen3-8B production path is approximately 55 tok/s at stable
peak. The current Qwen3.8-27B native GPU path measures approximately 3.36
tok/s for TG128 at stable peak and remains a bring-up/performance baseline.

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
