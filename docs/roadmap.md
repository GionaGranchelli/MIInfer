# MIInfer Roadmap

This document preserves historical milestone plans while defining the canonical
post-v0.2 direction. The **Current Released Baseline** and **Canonical Forward
Roadmap** below are authoritative; older milestone plans remain history and do
not authorize new work.

---

# Current Status

**Current released baseline: v0.2.0.** The qualified source is
[`94fad71`](release-v0.2.0.md), and the release archive SHA-256 is
`822fa647cec33efc34689630c0137870b5d61689e77a00b281bdba186a0f4519`.
Release qualification passed the pinned-runtime matrix and V2-0048B exact-
artifact gates. See the [release record](release-v0.2.0.md) and
[`V2-0048B evidence`](../experiments/V2-0048B-generation-completion-contract.md).

**Immediate project phase: M31 — Single-MI50 Agent Frontier Qualification.**
M29 and M30 have been integrated and evidenced on the M31-0000 baseline branch.
M31-0000 freezes the canonical source and environment only; it performs no
performance work. The qualification stages are:

```text
M31-0000 — Canonical baseline freeze
M31-0001 — Single-MI50 performance frontier
M31-0002 — Persistent-context qualification
M31-0003 — Canonical agent workload
M31-0004 — Stability and release gate
```

The next implementation goal is M31-0001 after M31-0000's clean-gate evidence
is recorded.

Historical M0–M28 and V2 records below remain valid history. Their old
“current,” “immediate next,” and milestone-forward statements are superseded by
the post-release roadmap in this document; their measurements and conclusions
are not rewritten.

---

# Canonical Forward Roadmap

```text
v0.2.x — Release / dogfood
        ↓
M29 — Persistent Context Architecture
        ↓
M30 — Agent Runtime Advantage
        ↓
M31 — Single-MI50 Agent Frontier Qualification
        ↓
V3 — Dual-MI50 Agent Engine
        ↓
Second-model / generalisation work
```

Milestone order is strategic direction, not authorization to implement the
entire sequence. Every stage begins through a focused, evidence-backed goal.
The immediate first engineering goal after M29-0000 is M29-0001, covering the
M29.0 long-context baseline and M29.1 physical-backing feasibility.

Before M29 work is split across the two available MI50 hosts, establish one
reproducible OCI-based gfx906 development environment on both the HP Z840 and
Machinist X99 systems. The container must pin the MIInfer userspace/toolchain
configuration and model/benchmark inputs while host-specific hardware and kernel
properties remain recorded evidence. Once both hosts pass the same smoke and
baseline checks, dependency-independent M29 goals may run in parallel. See
[`parallel-development.md`](parallel-development.md).

## M29 — Persistent Context Architecture

**Mission:** Build a placement-independent persistent context substrate,
qualify it for production 128K operation on one MI50, and make its logical
semantics reusable when physical execution later expands to two MI50s.

M29 distinguishes logical context semantics from physical KV backing:

```text
ContextSpace (identity, positions, pages, placement-independent semantics)
                         ↓
              Backing Strategy
          ┌──────────────┼──────────────┐
      contiguous      HIP-VMM       other qualified backing
          └──────────────┼──────────────┘
               optimized physical KV view
                         ↓
                      GQA kernel
```

`ContextSpace != HIP-VMM`. HIP-VMM is an experiment and one possible backing,
not a ContextSpace dependency. If it fails, choose another backing and continue
M29. Logical paging must not force a page-table lookup on every hot-path KV
access; a backing strategy may expose a pre-resolved physical view that keeps
kernel addressing equivalent to `base_ptr + offset`.

Stages (each requires its own scoped goal and gate):

```text
M29.0 — Current long-context baseline: 16K / 32K / 64K / 128K
M29.0A — Current gfx906 external-reference calibration
M29.1 — Physical-backing feasibility: HIP-VMM investigation
M29.2 — ContextSpace + logical pages
M29.3 — DeviceKvPool + DeviceKvShard
M29.4 — Two logical KV shards on one MI50
M29.5 — Optimized physical KV view / attention integration
M29.6 — Production 128K qualification
```

### M29.0A — Current gfx906 external-reference calibration

Before M29 changes context allocation or backing, refresh the strongest practical
external gfx906 baseline under the same MI50 conditions. The initial reference is
`kyuz0/mi50-gfx906-toolboxes` pinned at
`a708a2790fa51303e3d4f5af9e53c045177181a3`, using its ROCm 7.2.1
`llama.cpp` toolbox path. This is a bounded calibration lane, not a new
optimization campaign.

The first screen must use MIInfer's exact
`Qwen3.8-27B-Q4_K_M` model artifact/hash, the qualified 225 W and
1606/1000 MHz hardware policy, deterministic equivalent token inputs, and the
existing V2-0045 comparison points (P512/P1024/P2048/P4096/P8192 plus TG128)
where the external runtime can execute the same work. Record the toolbox commit,
llama.cpp submodule revision, ROCm/HIP compiler identity, gfx906 rocBLAS/Tensile
provenance, build flags, clocks, temperatures, power, VRAM, and raw samples.

Use the result only as an attribution gate:

- if the current toolbox reference stays within 3% of the pinned V2-0045
  upstream medians across equivalent cells and exposes no qualitative change,
  record the environment and stop;
- if an equivalent cell moves by more than 3% reproducibly, attribute the
  difference to source revision, ROCm/Tensile, ROCWMMA attention, build flags,
  or another measured cause before changing MIInfer;
- do not compare vLLM aggregate multi-request throughput directly with
  MIInfer's single-stream TG measurements;
- patched Triton, FlashAttention, vLLM, rocBLAS, RCCL, or Tensile components do
  not become MIInfer production dependencies merely because the toolbox ships
  them.

The detailed discovery and bounded implementation candidates are recorded in
[`gfx906-toolbox-reference.md`](gfx906-toolbox-reference.md).

Single-MI50 production qualification targets 128K. The logical architecture
may be designed for 256K capacity where inexpensive and technically useful;
this does not require 256K physical backing on one card. The later V3
dual-MI50 north star is 256K persistent physical context.

The physical ownership hierarchy is:

```text
ContextSpace → LogicalPageTable → PlacementPlan
             → DeviceKvShard[] → DeviceKvPool(device)
```

A logical page may map to one or more physical shards; each physical shard has
exactly one owning device. M29's one-card, two-shard experiment tests metadata,
allocation ownership, placement semantics, lifecycle, and kernel-facing views;
it is not evidence of dual-GPU performance. Logical page/context/session/
prefix/snapshot identity, forks, rollback, and COW metadata must not encode
physical GPU placement. KV-head placement may change without changing logical
identity.

`N=1` remains a first-class optimized topology; `N=2` extends it. Qualification
targets `N=1` and `N=2`; `N>2` is not a current product or qualification
target, and MIInfer will not build a generic distributed-KV framework. M29's
performance gate is no reproducible, statistically credible decode regression
greater than 1% attributable to the context architecture across the qualified
short/mid-context matrix, under equivalent hardware, thermal, workload,
software, and baseline conditions. Correctness is an independent mandatory
gate.

## M30 — Agent Runtime Advantage

M29 defines where and how context exists. M30 defines how sessions share,
branch, restore, and extend it over that substrate. Planned semantics include
persistent sessions, exact-prefix identity/reuse, recurrent-state reuse,
shared prefix pages, reference counting, copy-on-write, snapshots, fork,
rollback, append-only suffix prefill, and Tail-Replay / bounded suffix replay.
These are planned capabilities, not current implementation claims.

The primary KPI is total wall-clock time across realistic long-running coding
agent sessions, not only PP512 or TG128. Qualification should use roughly
20–50 turns with long persistent prefixes, tool calls/results, small appended
turns, repeated continuation, and fork/branch and rollback when supported. The
central workload is a large existing context plus a small new suffix whose
existing work should be reused rather than replayed.

## M31 — Single-MI50 Agent Frontier Qualification

M31 freezes and qualifies the completed N=1 generation before topology
changes. Evidence should cover cold/warm TTFT; 8K, 32K, 64K, and 128K context;
prefill, decode, VRAM, correctness, prefix reuse, suffix continuation,
snapshot/fork and rollback where available; realistic multi-turn coding-agent
wall clock; and operational stability. It answers: “How good is the finished
one-MI50 MIInfer agent runtime?” Only after M31 should the roadmap move to
dual-GPU execution.

M31 is intentionally qualification, not an optimization campaign. The frozen
N=1 result, including measured weaknesses, is the invariant for V3: dual-MI50
work must preserve N=1 correctness and must not materially regress the frozen
single-MI50 path.

## V3 — Dual-MI50 Agent Engine

V3's north star is two AMD Instinct MI50 32 GB devices serving one interactive
coding-agent session, initially with a Qwen 27B Q4_K_M-class model and
256K persistent context. `>=40 tok/s` decode is a stretch target, not a
promise; measured topology and correctness evidence may change the achievable
performance or execution strategy.

Provisional experiment progression:

```text
V3-0000 — Dual-MI50 topology qualification
V3-0001 — Physical DeviceKvShard placement
V3-0002 — Dual-device KV ownership qualification
V3-0003 — Candidate parallel attention execution
V3-0004 — Candidate parallel recurrent/GDN execution
V3-0005 — Cross-device communication/reduction
V3-0006 — Short-context dual-GPU decode qualification
V3-0007 — KV representation/capacity feasibility, if required
V3-0008 — 128K dual-GPU qualification
V3-0009 — 256K persistent-context qualification
V3-0010 — 256K agent-session qualification
V3-0011 — Dual-GPU performance frontier
```

GQA head parallelism and GDN tensor parallelism are hypotheses, not
architectural commitments. V3-0000 must measure PCIe topology, P2P support,
peer-read/write and bidirectional bandwidth, latency, synchronization,
IOMMU/ACS constraints, per-device VMM capability, and xGMI availability or
absence before selecting distributed execution. Do not invent or simulate
these results during M29/M30. Prefer device-local KV consumption; cross-device
hot-path traffic should primarily carry activations, partial results, or
reductions rather than repeated remote KV fetches. This preference does not
select the parallel algorithm.

## Second-model / generalisation work

Second-model work follows M31 and V3 in the immediate strategic order. Its
question remains whether MIInfer's architecture generalizes beyond the initial
model without becoming generic, but the nearer question is how far MIInfer can
push its intended agent workload on the targeted MI50 hardware.

## Cross-cutting — gfx906 reproducibility and distribution

The toolbox discovery also exposes a productization opportunity that is
independent of kernel research: make the qualified MIInfer runtime reproducible
for another MI50 owner without requiring reconstruction of the development
environment.

After M29.0A establishes the external reference environment, MIInfer may add a
provenance-pinned OCI/Podman distribution path for gfx906. It should package
only the userspace/runtime pieces MIInfer actually needs, preserve the exact
ROCm/toolchain and model/runtime provenance, expose hardware/environment checks,
and run a bounded `miinfer serve` smoke. Rebuilt gfx906 rocBLAS/Tensile/RCCL
artifacts remain optional reference or compatibility inputs unless controlled
evidence shows that MIInfer requires them.

This is packaging/reproducibility work, not authorization to replace MIInfer's
custom execution path with llama.cpp, vLLM, Triton, or a generic framework. It
does not change the M29 → M30 → M31 → V3 milestone order.

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

# Historical M26/M27 Execution Notes (superseded)

The following execution order was current before v0.2.0 release. It is retained
as historical planning context; it is not the current order and does not
authorize M26-E/M26-F work.

## Historical Execution Order

1. Preserve the clean, pushed M27 closure and the default-runtime M26 recovery
   result. Keep the experimental `213.837 tok/s` P512 candidate and M27 prefix
   cache frozen at their documented scopes.
2. Preserve EXP-0352's closure and its raw artifacts; do not use historical
   M8/M9 timing as a regression baseline for the interactive route.
3. Preserve the M26-CQ teacher-forced comparison and its `ROUTES_NOT_COMPARABLE`
   result; it establishes neither a correctness bug nor a timing result.
4. Base future M26-E/M26-F work on the qualified no-preset route. Do not repair
   the experimental interactive route solely to create an A/B comparison.

---

# Historical Immediate Next Milestone (superseded)

```text
M26-E/M26-F — Continue from the qualified no-preset route (not started)
```

The M26-E/M26-F proposal below was never started and is superseded by the
Canonical Forward Roadmap above. EXP-0352's historical and interactive timing contracts remain non-comparable.
EXP-0359 closes the current-route teacher-forced semantic investigation as
`ROUTES_NOT_COMPARABLE`: no accepted pairwise numerical tolerance exists, and
no timing claim was made. Preserve the qualified no-preset route as canonical;
M26-E/M26-F have not started. Do not reopen route attribution or begin M27
work based on this comparison.

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
