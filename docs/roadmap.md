# MIInfer Roadmap

This roadmap defines the staged development plan for MIInfer.

The project is intentionally structured around **evidence gates**.

Later milestones should not begin merely because earlier milestones are “mostly done.” Each stage exists to answer a specific technical question before the project accepts additional complexity.

---

# Current Status

**Current phase: M4 — First Correct Generation**

Immediate objective:

> Execute the minimum correct Qwen3-8B path using the closed M3 loader and
> static plan, then validate deterministic first-token generation.

Current work should focus on:

* embedding and normalization execution
* Q/K/V projection integration
* deterministic attention and residual execution
* first-token correctness against the pinned reference
* preserving the accepted kernel and benchmark controls

Do not start model serving, generic model support, speculative decoding, or multi-GPU work.

---

# Roadmap Principles

## Evidence before architecture

Do not design large runtime abstractions before benchmark and kernel work establishes what the runtime actually needs.

## Narrow before broad

Support one hardware target, one workload class, one model family, and one small set of execution paths before expanding.

## Correctness before speed

Performance work must retain numerical and model-level correctness.

## Benchmark before claim

No milestone is complete because code exists.

Milestones complete when their technical question has been answered with reproducible evidence.

---

# M0 — Baseline and Project Bootstrap

## Goal

Create the environment required to perform credible MI50 inference research.

M0 does **not** implement LLM inference.

## Questions

M0 should answer:

* Can the development environment reliably compile HIP code for gfx906?
* Can MI50 hardware state be captured before benchmark runs?
* What implementation will serve as the initial external performance baseline?
* Can benchmark results be reproduced across repeated runs?
* Can correctness reference results be generated independently of candidate kernels?

## Deliverables

### Repository

* `AGENTS.md`
* `README.md`
* `LICENSE`
* CMake build
* canonical CMake presets
* formatting configuration
* test structure
* benchmark structure
* experiment template
* documentation structure

### Hardware environment

Document:

* GPU model
* gfx architecture
* ROCm version
* HIP compiler version
* kernel version
* relevant runtime configuration
* GPU clock reporting
* HBM clock reporting where available
* temperature
* power
* VRAM state

### Reference runtime

Establish the strongest reproducible practical gfx906 inference baseline available.

Current preferred direction:

* use a dedicated gfx906 llama.cpp reference repository/configuration
* preserve its exact commit and build configuration
* record benchmark commands
* do not modify it as part of MIInfer implementation

### Benchmark protocol

Define:

* warm-up behavior
* repeated runs
* interleaved A/B methodology
* hardware-state validation
* raw result storage
* aggregate statistics
* contamination rules

## Exit criteria

M0 is complete when:

* a trivial HIP program runs successfully on gfx906
* the build can target gfx906 reproducibly
* environment capture works
* benchmark runs produce machine-readable output
* the external reference implementation can be built and run
* baseline performance is recorded
* benchmark methodology is documented

## Non-goals

Do not implement:

* model loader
* tokenizer
* inference loop
* attention
* HTTP server
* speculative decoding
* multi-GPU support

---

# M1 — Kernel Laboratory

## Goal

Build the infrastructure required to develop and evaluate gfx906-native kernels independently of a complete LLM runtime.

## Questions

M1 should answer:

* Can candidate kernels be compared against trusted reference implementations?
* Can kernel timing be measured with sufficiently low noise?
* Can effective bandwidth and execution characteristics be observed?
* Which representative LLM kernel shapes should become permanent benchmark cases?

## Initial kernel areas

Start with operations that are relevant to low-batch inference:

* FP16 matrix-vector multiplication
* quantized matrix-vector multiplication
* quantization of activations
* reductions
* RMSNorm

Do not begin with FlashAttention unless profiling evidence justifies doing so.

## Benchmark harness

The harness should support:

* deterministic input generation
* CPU/reference execution
* GPU execution
* warm-up
* repeated timing
* numerical comparison
* configurable matrix dimensions
* machine-readable result export

## Initial representative shapes

Representative shapes should come from the selected target model.

Avoid synthetic dimensions that do not occur in real execution unless they test a specific architectural property.

## Exit criteria

M1 is complete when:

* GPU kernels can be benchmarked independently
* CPU/reference correctness comparisons exist
* representative decode shapes are recorded
* raw benchmark output is reproducible
* at least one baseline kernel family is established

---

# M2 — Prove Specialization

## Goal

Determine whether gfx906-specific specialization provides enough performance benefit to justify an independent runtime.

M2 is the project's first major **go/no-go gate**.

## Core question

> Can dedicated MI50/gfx906 kernels materially outperform the strongest existing gfx906 implementation for important target-model operations?

## Candidate research areas

### Wave execution

Compare:

* Wave64 cooperative execution
* logical half-wave groups
* alternative row/work distribution

### gfx906 primitives

Investigate where appropriate:

* DPP operations
* `ds_swizzle`
* lane broadcast
* packed integer dot products
* native math instructions

### Quantized execution

Candidate paths:

* Q4 × Q8
* Q6 × Q8
* Q8 × Q8
* MXFP4 if relevant to target model
* FP16 operands with FP32 accumulation

### Weight layout

Compare:

* canonical model/quant layout
* kernel-native repacked layout
* interleaved scales
* pre-transposed/block-oriented storage

### Memory behavior

Investigate:

* global load patterns
* alignment
* coalescing
* cache behavior
* LDS usage
* VGPR pressure
* explicit prefetch only where evidence supports it

### Kernel configuration

Study:

* workgroup sizes
* logical wave width
* tile size
* launch bounds
* occupancy constraints
* register/LDS trade-offs

## Required experimental discipline

Every candidate optimization must receive:

* experiment ID
* hypothesis
* baseline
* candidate
* correctness result
* repeated benchmark
* hardware-state record
* KEEP / REJECT / RETEST decision

## Success criteria

There is deliberately no single universal percentage threshold.

However, M2 should demonstrate at least one of:

* meaningful kernel latency reduction on a major decode bottleneck
* meaningful effective-bandwidth improvement
* meaningful prompt-processing improvement
* meaningful generation-throughput improvement in an isolated model-level prototype

The improvement must survive repeated controlled measurement.

A tiny synthetic-kernel improvement that has no plausible model-level impact does not satisfy M2.

## Go / no-go decision

### GO

Continue to M3 if specialization demonstrates credible superiority against the
strongest relevant gfx906 implementation on important target-model
operations. An isolated, correctness-valid kernel comparison is sufficient;
a complete MIInfer runtime is not required for the M2 gate.

### REASSESS

Pause runtime work if:

* optimized kernels consistently match existing gfx906 implementations
* gains occur only in irrelevant shapes
* correctness requires unacceptable compromises
* implementation complexity substantially exceeds realistic benefit

A reassessment is a valid project result.

EXP-0009 passed this gate: the accepted shape-specialized MIInfer geometry is
competitive with or faster than the pinned gfx906 MMVQ path across the seven
core projection shapes, with D retaining the already-accepted EXP-0007
geometry. M2 is therefore `GO`, and the project has advanced to M3.

---

# M3 — Minimal Runtime

## Goal

Build only enough runtime infrastructure to execute one explicitly supported model architecture.

Do not build a general-purpose inference framework.

## Core runtime responsibilities

Implement:

* model metadata loading
* tensor discovery
* supported-model validation
* GPU memory allocation
* weight loading
* optional one-time repacking
* static buffer lifetime planning
* static kernel selection
* execution-plan creation

## Architectural rule

Prefer:

```text
load once
plan once
allocate once
select once
execute repeatedly
```

Avoid repeated hot-path decisions.

## Model scope

Support exactly the selected initial model/configuration required by the project roadmap.

Unsupported configurations should fail explicitly.

Do not silently fall back to generic code.

## Quantization scope

Support only the quantization required by the selected initial target.

Adding additional formats belongs later unless they are required for M2/M3 comparison.

## Exit criteria

M3 is complete when:

* the supported model is recognized
* required tensors load correctly
* tensor dimensions are validated
* weights fit within the planned memory layout
* static execution plan can be constructed
* unsupported models/configurations fail clearly

M3 was closed on 2026-08-30 by the pinned Qwen3-8B Q4_0 physical-MI50
acceptance. The implementation recognizes the model, validates its 399
tensors, uploads all immutable weights into a 4.77 GB arena, verifies bounded
device-byte samples, and constructs the static buffer/kernel plan. It does not
execute the transformer yet.

No meaningful generated text is required yet.

---

# M4 — First Correct End-to-End Generation

## Goal

Generate correct tokens from the selected target model on one MI50.

Performance is secondary to correctness during the first pass.

## Required execution pieces

Depending on the selected architecture:

* token embedding
* RMSNorm
* Q/K/V projections
* RoPE
* attention
* output projection
* residual connections
* FFN or MoE
* final norm
* logits projection
* sampling/greedy selection

## Initial inference mode

Prefer:

* single sequence
* batch 1
* deterministic / greedy decoding
* fixed small context
* text-only

Avoid adding serving infrastructure.

## Correctness validation

Compare against a trusted reference implementation.

Validate:

* logits where practical
* selected next token
* short generation
* repeated generation
* numerical stability

## Exit criteria

M4 is complete when:

* the target model generates coherent output
* deterministic generation matches or stays within an explicitly accepted reference tolerance
* no NaN/Inf behavior occurs
* repeated runs are stable
* basic context growth works correctly

---

# M5 — Beat the Reference

## Goal

Determine whether MIInfer's end-to-end architecture delivers measurable advantages over the strongest reproducible gfx906 baseline.

This milestone answers the central project hypothesis.

## Comparison dimensions

At minimum measure:

### Prompt processing

* PP512
* PP2048
* PP8192 where practical

### Decode

* TG128
* TG512
* TG1024

### Context regimes

At least:

* short
* medium
* long enough to expose attention/KV effects

### System metrics

Record:

* VRAM
* power
* temperature
* clocks
* tokens per joule where practical
* time to first token

## Comparison rules

The implementations must use equivalent:

* model
* quantization
* context
* sampling
* power limit
* GPU state

If the representations differ because MIInfer uses a custom packed layout, document the difference explicitly.

## Exit criteria

M5 is complete when:

* reproducible end-to-end comparison exists
* correctness remains acceptable
* performance differences are explained by profiling
* major bottlenecks are identified
* project hypothesis receives an explicit result

Possible outcomes:

### H1 supported

MIInfer demonstrates meaningful advantage.

Proceed to M6.

### Mixed result

Some workloads improve while others regress.

Document the trade-offs and decide whether architecture changes can realistically improve the result.

### H0 supported

The specialized runtime provides no meaningful benefit.

Do not hide this result.

Reassess project continuation.

---

# M6 — Runtime Specialization

## Goal

Once end-to-end viability is demonstrated, optimize runtime-level overhead and data movement.

## Candidate work

### Native weight packing

Introduce a model artifact or loading-time transformation optimized for MI50 kernel consumption.

Potential objectives:

* contiguous quant values
* separate/interleaved scale planes
* exact block ordering
* alignment for vectorized loads

### Activation reuse

Avoid repeated quantization or transformation when multiple projections consume the same activation.

### Static memory plan

Eliminate unnecessary runtime allocation.

### Operation fusion

Fuse operations only where profiling shows a worthwhile launch/memory benefit.

Potential examples:

* norm + quantization
* bias/scale epilogues
* activation functions with projection stages

### HIP graph capture

Measure decode execution with:

```text
ordinary dispatch
vs
captured/replayed execution
```

Graph capture must remain correctness-safe.

### Kernel specialization

Replace remaining generic/fallback kernels where profiling identifies significant cost.

## Exit criteria

M6 is complete when:

* runtime overhead has been profiled
* major avoidable hot-path allocations are removed
* graph capture has been evaluated
* packed layout strategy is decided
* end-to-end performance improves or experiments are explicitly rejected

---

# M7 — Expansion

M7 begins only after the original MI50 target has demonstrated sufficient value.

Expansion should happen incrementally.

## Potential areas

### Additional quantization

Examples:

* alternative Q4 format
* Q6
* Q8
* MXFP4

Only add formats with a clear use case.

### Second model

Supporting a second model tests whether MIInfer's architecture can generalize without becoming generic.

Prefer a model that answers a specific question.

For example:

* dense vs MoE
* different head dimension
* different FFN dimensions

### Long-context specialization

Investigate:

* KV precision
* Q8 K
* FP16 V
* quantized KV
* attention bandwidth
* context-dependent kernel selection

### Speculative decoding / MTP

Only after baseline decode is well understood.

Investigate:

* draft depth
* adaptive speculation
* acceptance rate
* verification cost
* context-dependent payoff

### Serving

Optional:

* CLI
* minimal HTTP server
* OpenAI-compatible endpoint

Serving must not dictate the core runtime architecture.

---

# Deferred / Explicitly Out of Scope

The following should remain deferred unless the roadmap is explicitly revised:

* CUDA
* generic AMD support
* RDNA
* MI200 / MI300
* Intel GPUs
* CPU optimization
* Windows
* macOS
* training
* fine-tuning
* multimodal models
* arbitrary GGUF support
* arbitrary Hugging Face model support
* tensor parallelism
* pipeline parallelism
* distributed inference
* continuous batching
* high-concurrency serving

---

# Milestone Dependencies

```text
M0
│
▼
M1
│
▼
M2 ──────────────┐
│                │
│ GO             │ REASSESS
▼                ▼
M3             project decision
│
▼
M4
│
▼
M5 ──────────────┐
│                │
│ advantage      │ no advantage
▼                ▼
M6             reassess
│
▼
M7
```

---

# Current Execution Order

The project has passed M2 and M3, and M4-A is complete. Work should proceed
from the accepted model plan and layer-state correctness evidence:

1. Preserve the M0 platform contract and pinned reference.
2. Maintain the accepted FP16 and Q4/Q8 kernel controls.
3. Localize M4-B full-depth numerical drift with teacher-forced replay.
4. Resolve the layer-6 FFN projection precision discontinuity without
   widening numerical tolerances.
5. Preserve the proven exact Q4_0 × Q8 zero-point correction for Down and
   isolate the remaining upstream/late-depth numerical drift.
6. Establish external layer-6 trace repeatability and use exact-Q8
   F16/F32 projection probes to separate GPU precision boundaries from the
   remaining host/reference contract.
7. Canonicalize the full-depth external CPU fixture, verify host replay, and
   validate the minimum causal-path precision correction before proceeding
   to token generation.
8. Use the M4-B9 terminal layer-35 internal trace to resolve the shared
   host/reference FFN-tail numerical contract before changing production
   precision.
9. Use M4-B10 host Gate/Up hybrid SwiGLU attribution to identify which
   projection contract feeds the remaining terminal-layer discrepancy, then
   compare pinned Q8 quantization and accumulation semantics.
10. Use M4-B11 Q8 identity and external-conditioned projection replay to
    locate the earlier FFN-input/attention-output contract that produces the
    differing terminal-layer `ffn_norm`.
11. Apply the proven attention-output FP16 boundary to production host and
    MI50 execution, then preserve it as the accepted M4-B14 contract.
12. Localize remaining full-forward host drift with sequential-versus-isolated
    layer traces and verify the full-forward entry point has no independent
    orchestration divergence (M4-B15).
13. Trace layer-0 against the independent 28-checkpoint fixture and reject
    an unsupported extra layer-output FP16 materialization (M4-B16).
14. Use external attention-output injection to separate the causal V/attention
    perturbation from downstream O/FFN variance (M4-B17).
15. Test exact-Q8 V input/output precision policies and replay each through
    GQA, attention materialization, and the causal O/FFN tail (M4-B18).

Do not broaden M4 into a general-purpose runtime.

---

# Immediate Next Milestone

The current milestone is:

```text
M4-B — full-depth numerical validation
```

M4-B8 established a canonical full-depth CPU fixture from two byte-identical
runs of the pinned reference with explicit `-t 24 -tb 24` settings. M4-B9
added a terminal layer-35 internal trace from the same independent reference
and proved the external layer-output tensor is identical to the input of
final RMSNorm. M4-B10 added host Gate/Up hybrid attribution: external Gate
plus Up is within `7.62939e-06`, while replacing either input with the host
projection reproduces most of the `0.105103` discrepancy. M4-B11 then
replayed the pinned x86 AVX Q8 contract and Gate/Up accumulation with
external `ffn_norm`; all Q8 blocks matched and Gate/Up were within
`7.62939e-06`/`1.52588e-05`. The remaining failure therefore enters before
`ffn_norm`, and M4-B is still open. M4-B12 then showed that externally
conditioned O and residual replay are within `9.15527e-05`/`3.05176e-05`,
while all tested RMSNorm reductions produce a layer-35 tail within
`0.000488281`. The remaining normal-path difference therefore enters
upstream in the attention output that feeds O. M4-B13 then showed that V and
GQA replay are exact or near-exact, while the external attention output is
exactly `FP16(expanded V)`; applying that boundary reduces the layer-35 tail
to `0.000488281`. The remaining precision hypothesis is the attention-output
materialization before O, and M4-B remains open. M4-B14 applied that boundary
to production host and MI50 execution. Focused layer-35 host parity and the
full MI50 GPU gate pass, but host full-forward parity first fails at layer 2
(`max_abs=0.117966`). M4-B15 shows the host full-forward entry point is
bitwise identical to a reconstructed sequential chain: the first inherited
sequential difference is layer 1 input, and layer 2 is the first strict
threshold crossing. M4-B remains open without any tolerance widening. Exact
experiment ordering may change only when supported by the resulting evidence.
M4-B16 then found the first layer-0 mismatch at `q_projection`, while the
position-zero causal V/FFN path drifted gradually; an additional layer-output
FP16 round-trip worsened the error and was not accepted.
M4-B17 then showed that external attention injection reduces host layer-0
error from `0.00548154` to `1.90735e-06` and MI50 error from `0.00704432` to
`0.00282186`, proving the causal V/attention perturbation dominates.
M4-B18 then showed that F32->Q8Exact->F32 gives near-exact local V parity
(`1.90735e-06`), but downstream materialization and quantized O sensitivity
still leave approximately `0.204956` layer-output error; no production
precision change is accepted yet.

---

# Roadmap Change Policy

The roadmap is not immutable.

However, changes should be driven by evidence.

When changing milestone scope:

1. state what new evidence motivated the change
2. identify which assumption changed
3. update `docs/current-state.md`
4. update relevant decision records
5. preserve previous experiment results

Avoid roadmap changes based solely on implementation convenience.

---

# Success Definition

MIInfer does not succeed merely by running an LLM on an MI50.

Existing projects already do that.

The project succeeds if it produces a rigorous answer to:

> Does a runtime intentionally designed around gfx906 and a narrow model target provide meaningful advantages over the strongest generic-runtime implementations available for the same hardware?

That answer must come from reproducible measurement.
