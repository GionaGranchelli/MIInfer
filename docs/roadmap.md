# MIInfer Roadmap

This roadmap defines the staged development plan for MIInfer.

The project is intentionally structured around **evidence gates**.

Later milestones should not begin merely because earlier milestones are “mostly done.” Each stage exists to answer a specific technical question before the project accepts additional complexity.

---

# Current Status

**Current phase: M6-B44 — post-B41 profile and Q4 split-K rejection**

Immediate objective:

> Select the next bounded whole-token experiment from the post-B41 profile;
> retain decoded Q4_K metadata staging and the existing Q6_K×Q8_1 LM head.

M5 closed with a reproducible local optimization result, but whole-runtime
parity with the strongest gfx906 llama.cpp control was not demonstrated. See
`experiments/EXP-0041-m5c15-optimization-closure-parity-gate.md`. M6-A0 audited
the Qwen3.8-27B artifact, M6-A1 provides a pinned llama.cpp reference fixture,
and M6-A2 maps the reusable and missing projection contracts. See
`experiments/EXP-0043-m6a1-qwen38-reference-fixture.md` and
`experiments/EXP-0044-m6a2-qwen38-projection-compatibility.md`.

Current work should focus on:

* preserving the M6-B30 transposed recurrent-state layout and its row-major
  control flag for reproducible A/B comparisons
* refreshing the post-B30 profile before selecting the next optimization
* preserving EXP-0091's margin-aware observable closure: exact teacher-forced
  agreement remains a `62/64` diagnostic, while P2/P12 are low-margin,
  top-5-preserving flips
* not reopening the cleared L3/P2 or Q5_K investigations without new evidence
* retaining the accepted external state contract and the strict `0.05`
  recurrent-state check as diagnostic-only
* preserving GPU-resident recurrent/KV state and zero steady-state allocations
* state fingerprints, reset/replay checks, and external checkpoints through P64
* per-layer telemetry and the evolving VRAM ledger
* preserving the M6-A1 external correctness authority
* keeping the old qwen3 production path unchanged
* retaining EXP-0092's A28 result: 16/64/128 native runs pass exact replay,
  with zero decode-loop allocations and stable device usage
* retaining EXP-0122's result: native TG64/TG128 improve by about 3.4%/3.6%
  with unchanged logical state behavior and no additional VRAM
* retaining EXP-0124's result: native TG64/TG128 improve by about 1.71%/1.63%
  with the transposed state layout and no-decay-store recurrent update
* retaining EXP-0125's profile: total GPU work is about 84.6–84.7 ms/token and
  recurrent FFN projection work remains the largest repeated family
* retaining EXP-0126's rejection: fused SiLU/Q8_1 passed external correctness
  but was throughput-neutral at TG64/TG128 and was removed
* retaining EXP-0127's result: Q4_K×Q8_1 LDS activation reuse improves native
  TG64/TG128 by about 2.98%/3.01% with unchanged device usage
* retaining EXP-0128's profile: total GPU work is 82.1223 ms/token and long-K
  Q4_K×Q8_1 FFN Down remains the largest repeated projection family

Do not treat CPU hidden-state identity through all 36 layers as a universal
GPU requirement. B23 characterized the pinned external implementation's
distinct CPU Q4_0×Q8_0 and single-token gfx906 Q8_1/MMVQ execution contracts;
B24 closed M4-B using a measured final-output and behavioral envelope while
retaining strict semantic invariants.

Do not start model serving, generic model support, speculative decoding, or
multi-GPU work. Do not report M6-B1 performance until a real qwen35 HIP path
exists; see `experiments/EXP-0051-m6b1-qwen38-miinfer-gpu-readiness.md`.

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
16. Compare attention materialization, Q8 codes, dequantized values, and an
    external-attention GPU control before changing V precision (M4-B19).
17. Characterize O/FFN GPU arithmetic with identical external inputs and Q8
    metadata before defining a full-layer backend-equivalence policy (M4-B20).
18. Test the minimum full-model F32-output policy and compare it with the
    combined F32 input/output diagnostic policy (M4-B21).
19. Compare the canonical external CPU trace with the independent offloaded
    gfx906 trace, and characterize backend-specific full-depth variance before
    changing precision or acceptance thresholds (M4-B22).
20. Define the measured MI50 final-output/behavioral envelope and run the
    non-vacuous Debug/Release physical acceptance gate (M4-B24).
21. M4-C1: execute the accepted 36-layer path incrementally with persistent
    per-layer KV state and produce the first deterministic generated token.
22. M4-C2: validate four to sixteen deterministic greedy decode steps before
    adding tokenizer or sampling behavior.

Do not broaden M4 into a general-purpose runtime.

---

# Immediate Next Milestone

The current milestone is:

```text
M4-C3 — tokenizer/detokenizer and minimal text-facing greedy decode
```

M4-B is closed by B24. The canonical physical gate now runs the real pinned
model through MI50 Debug and Release, compares final norm and logits against
the independently measured external CPU↔gfx906 envelope, verifies finite and
bitwise-deterministic repeated output, and requires matching argmax and
baseline top-5 overlap. M4-A's strict four-position layer-0/KV-cache gates
remain prerequisite evidence. The detailed B8–B24 record is preserved in
[`m4b-forward.md`](m4b-forward.md).

M4-C1 is closed. It combines the accepted 36-layer single-token path with
the proven incremental KV-cache semantics, selects token `8` from prompt token
`14990`, and consumes that generated token at position `1` using persistent
per-layer state. Debug and Release physical acceptance pass, as do the
artifact-free `18/18` regression suites. M4-C2 validated the pinned eight-
token explicit-ID greedy sequence through persistent per-layer KV state.
Release matches all eight reference IDs and deterministic replay. The
unoptimized Debug build remains a finite/cache/determinism diagnostic and
reports `419` where the independent reference selects `470` at position 3.
The fixed-prefix Debug/Release dump localizes the first output difference to
layer 20 during position 1; serialized Debug is unchanged and RelWithDebInfo
follows Release, pointing to unoptimized HIP code generation rather than a
cache-ordering race. Optimized-HIP Debug (`-O2 -g`) matches Release. M4-C2 is
closed; tokenizer, sampling, serving, batching, and performance work remain
out of scope except for the minimal tokenizer/detokenizer work in M4-C3.

M4-C3 is closed as a narrow text-facing layer. `Qwen3Tokenizer` consumes
the pinned GGUF `gpt2`/`qwen2` vocabulary and merges, while
`miinfer-qwen3-generate` runs prompt tokenization, persistent MI50 greedy
decode, EOS handling, and detokenization. Its real-model Release acceptance
uses prompt `hello` and the pinned eight-token continuation. Sampling,
streaming, and batching remain deferred. The next task is to establish the
first reproducible MI50 prefill/decode baseline before optimization.

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
`ffn_norm`, and M4-B was still open at that point. M4-B12 then showed that externally
conditioned O and residual replay are within `9.15527e-05`/`3.05176e-05`,
while all tested RMSNorm reductions produce a layer-35 tail within
`0.000488281`. The remaining normal-path difference therefore enters
upstream in the attention output that feeds O. M4-B13 then showed that V and
GQA replay are exact or near-exact, while the external attention output is
exactly `FP16(expanded V)`; applying that boundary reduces the layer-35 tail
to `0.000488281`. The remaining precision hypothesis is the attention-output
materialization before O, and M4-B remained open at that point. M4-B14
applied that boundary
to production host and MI50 execution. Focused layer-35 host parity and the
full MI50 GPU gate pass, but host full-forward parity first fails at layer 2
(`max_abs=0.117966`). M4-B15 shows the host full-forward entry point is
bitwise identical to a reconstructed sequential chain: the first inherited
sequential difference is layer 1 input, and layer 2 is the first strict
threshold crossing. M4-B remained open without any tolerance widening at
that point. Exact
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
M4-B19 then showed that external attention quantizes bitwise-identically to the
host Q8 contract and that the external-attention GPU control itself retains
`0.204956` layer-35 error, identifying a downstream GPU arithmetic floor.
M4-B20 then showed that identical-input F32-output GPU projections remain
within `0.000244141` of the external CPU outputs, while F16-output controls
produce materially larger direct errors; full-layer parity remains open.
M4-B21 then rejected output-only F32 as a complete policy: it reaches
`21.8325` layer-35 error, while the combined diagnostic policy reaches
`12.5605` and `0.131546` logits error. No production precision change is
accepted.
M4-B22 then added a threshold-free trace comparator and compared the pinned
external CPU and offloaded gfx906 traces. They differ by `121.013` at layer 35,
`0.811852` at final norm, and `0.488072` at logits while both select argmax
`8`. M4-B23 explained that split as distinct backend contracts. M4-B24 then
made the measured final-output/behavioral envelope the MI50 acceptance gate:
Debug and Release are finite and deterministic, MIInfer stays within the
external final-norm/logit split, top-5 overlap is at least the measured
baseline, and argmax remains `8`. M4-B is closed; M4-C is next.

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
