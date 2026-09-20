# MIInfer Roadmap

This roadmap defines the staged development plan for MIInfer.

The project is intentionally structured around **evidence gates**.

Later milestones should not begin merely because earlier milestones are “mostly done.” Each stage exists to answer a specific technical question before the project accepts additional complexity.

---

# Current Status

**Current phase: M26 — Decode Route Attribution**

M26-B historical decode-floor contract recovery is closed as non-comparable in
[`EXP-0352`](../experiments/EXP-0352-m26b-historical-decode-floor-differential.md).
The historical benchmark starts from fixture state at decode position 0 and
synchronizes/copies one token each step. The current M25 interactive result
starts after a 512-token prefill and uses queued graph replay with bulk token
transfer. These are different workloads and timing boundaries, so the
historical `31–33 ms/token` result is not a regression baseline for the
interactive `57–59 ms/token` route. The same-fixture current-code control shows
no shared-pipeline regression; its fresh run had isolated clock dips and is
retained as diagnostic evidence only.

The separate current layer-major versus legacy P512 decode premium remains
unattributed. The qualified no-preset runtime result `32.2948 ms/token` meets
the `<=33 ms/token` recovery screen but does not qualify the interactive route.
Continue only with a matched diagnostic for that current-runtime route delta;
do not select a decode kernel from the historical comparison or broad stage
profiles. See EXP-0351 and EXP-0352.

M27 cold P512 work is stopped. The current control is `205.897 tok/s`; the
`213.837 tok/s` attention decode-reuse candidate remains experimental because
its cold-prefill effect is unexplained. Equivalent contract attribution found
`+16.029 ms` for recurrent work and `+101.979 ms` for attention, but the
technically motivated FP16 Q/K candidate failed the exact-context screen at
`-1.49 ms` median. Do not promote the candidate or reopen cold tuning without a
new measured hypothesis. See EXP-0355 and EXP-0356.

M27 exact prefix reuse is frozen at the current opt-in scope: sparse exact
B512 checkpoints, longest-prefix radix lookup, and a 3 GiB payload cap. Real
Pi read/edit/read flows reused the complete cached prefix at ~3.7K, ~7.7K, and
~15K conversation sizes. Do not add persistence, cross-session policy,
additional cache tiers, or NVMe without new evidence. See EXP-0350 and EXP-0357.

Current-runtime decode route attribution remains the highest project priority.
Keep the default, qualified prefill, and experimental interactive execution
contracts distinct, and preserve the M26-B non-comparability finding and M27
evidence.

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

## M5-C15 — Optimization closure and parity decision gate

M5-C15 closes the optimization campaign and records the accepted/rejected
experiments, current throughput/context behavior, eliminated causes, and
remaining contract limitations. It explicitly gates the next phase:

* **Path A:** preserve the current trajectory and move to robustness,
  usability, longer-context, model-support, and release work.
* **Path B:** begin M6-A reference-correct execution-contract exploration,
  where alternate representations and precision boundaries are judged against
  a pinned external Qwen3 correctness envelope.

Do not start M6-A experiments until Path B is explicitly selected.

---

# M6-A — Reference-correct execution-contract exploration

M6-A is entered only if Path B is selected at M5-C15.

## Goal

Determine whether alternate execution contracts can improve MIInfer toward
llama.cpp-class performance while remaining correct for the pinned Qwen3 model.

## Contract

Candidates are no longer required to reproduce current MIInfer bytes or its
exact deterministic trajectory. They must instead satisfy a pinned external
Qwen3 correctness envelope, model semantics, and behavioral generation tests.
llama.cpp arithmetic is a comparison/reference implementation, not a new
golden tensor contract.

## Candidate areas

* Q6_K × Q8_1 LM-head input representation
* alternate activation representations
* precision and materialization boundaries
* reference-validated fusion
* alternate attention reduction structure
* graph-level representation planning

Each candidate still requires an isolated correctness result, reproducible
benchmark, VRAM/latency accounting, and an explicit KEEP/REJECT decision.

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

1. Preserve the clean, pushed M27 closure and the default-runtime M26 recovery
   result. Keep the experimental `213.837 tok/s` P512 candidate and M27 prefix
   cache frozen at their documented scopes.
2. Preserve EXP-0352's closure and its raw artifacts; do not use historical
   M8/M9 timing as a regression baseline for the interactive route.
3. If decode performance work resumes, measure the current layer-major versus
   legacy P512/TG128 route delta under interleaved A/B with continuous clock
   telemetry and per-family attribution.
4. Consider a decode hypothesis only when that current-route bottleneck is
   measured. No kernel has been selected by the completed M26-B comparison.

---

# Immediate Next Milestone

```text
M26 — Current-runtime route attribution (diagnostic only)
```

EXP-0352 B1–B5 is closed: the historical and interactive contracts are
non-comparable, and no historical kernel regression is claimed. The current
no-preset P512/TG128 route has a qualified `32.2948 ms/token` result; the
experimental layer-major route remains around `57–59 ms/token`. Recover the
current-runtime route differential with matched workload and clock telemetry
before proposing decode changes. The existing recovery result is not
authorization for sub-30 ms tuning.

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
M6-A27.9 found a real per-block Q5_K × Q8_K arithmetic-contract mismatch and
fixed it with integer partial accumulation plus reference scale/minimum
handling. The selected row's block-sum error fell to `0`, and external-gated
projection error fell to `9.53674e-7`; Release CTest is 20/20. The fix now
passes the L0–L2 P2 boundaries at roundoff, but the unchanged 64-layer retest
still has P2 `1318 → 1044` and P12 `1044 → 1459` (62/64 teacher-forced
agreement). The first visible P2 discrepancy is now L3 (`0.00255775`). See
`experiments/EXP-0090-m6a279-qwen35-a27-observable-retest.md` and
`experiments/EXP-0089-m6a279-qwen35-l0-q5k-block-contract.md`.

M6-A27.8 cleared the L0 Q8_K activation representation: the external-gated
P2 replay matches the pinned llama.cpp Q8_K reference byte-for-byte across all
7,008 bytes. The remaining `ssm_out` discrepancy is downstream in the Q5_K ×
Q8_K projection/accumulation contract. See
`experiments/EXP-0088-m6a278-qwen35-l0-q8k-contract.md`. No production change
was made.
