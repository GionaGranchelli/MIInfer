# Performance Research Strategy

This document defines the performance-research method MIInfer should use after
M26-C. It is a change in decision process, not a claim that a new optimization
has already been found.

The immediate context is
[`EXP-0358`](../experiments/EXP-0358-m26c-current-decode-route-differential.md):
the current layer-major and legacy decode routes must first be compared from an
equivalent semantic state. Historical M8/M9 decode timings remain
non-comparable to the current interactive route, as recorded in
[`EXP-0352`](../experiments/EXP-0352-m26b-historical-decode-floor-differential.md).

---

## 1. Mental model

MIInfer should separate three different kinds of performance work.

### Runtime recovery

Remove costs that are introduced by the execution system rather than required
by the model:

- host synchronization
- repeated H2D/D2H traffic
- graph setup or update work on the token critical path
- state copies
- avoidable allocation
- route-selection overhead
- CPU-mediated per-token orchestration

This class can produce large gains when the runtime is paying for work outside
the model computation.

### Operator improvement

Improve the implementation of required operators:

- GEMV/GEMM kernels
- quantization
- RMSNorm
- attention
- recurrent/GDN state updates
- FFN projections
- layout and LDS/register mapping
- fusion of already-required operations

This class should be assumed to have diminishing returns once the measured path
approaches the hardware roofline.

### Algorithmic reformulation

Change how the same model mathematics is evaluated without changing the model
contract:

- chunkwise recurrence
- algebraic reassociation
- persistent execution
- alternative state representations
- cross-operator scheduling
- prefill/decode-specific execution plans

This is the only class that should be expected to produce large gains after a
well-optimized operator path has approached memory or compute limits. Large
implementation effort requires a modeled ceiling before it begins.

---

## 2. Headroom before implementation

The default research loop is now:

```text
measure
  ↓
build cost model
  ↓
estimate physical / algorithmic floor
  ↓
quantify removable end-to-end cost
  ↓
authorize or reject experiment
  ↓
implement only if the ceiling is material
```

Do not start from "this kernel looks inefficient."

For every proposed performance experiment, record:

1. current end-to-end cost
2. cost of the targeted component
3. bytes moved and effective bandwidth where relevant
4. arithmetic work and observed compute utilization where relevant
5. launch, synchronization, and host-side contribution
6. plausible lower bound
7. maximum removable end-to-end cost
8. correctness risk
9. implementation cost
10. explicit pass/fail gate

A local kernel speedup is not sufficient evidence if the whole-path ceiling is
small.

---

## 3. Default authorization thresholds

These are research defaults, not universal laws.

### Decode/runtime work

Do not authorize a substantial runtime redesign solely for performance unless
the measured path identifies at least **3 ms/token** of runtime/execution cost
that the redesign can plausibly remove.

A redesign may still be justified for correctness, serving semantics, memory
safety, or required product behavior, but that must be stated separately from
a performance claim.

### Kernel micro-optimization

Do not authorize a non-trivial kernel campaign when the estimated maximum
whole-path gain is below approximately **2%** unless it removes a correctness
issue or unlocks another measured optimization.

### Algorithmic reformulation

Do not authorize a large prefill or recurrent reformulation when the modeled
whole-path ceiling is below approximately **5%**. Prefer research targets with
clear double-digit modeled headroom.

The purpose of these thresholds is to stop repeated experiments whose best
possible outcome is already known to be marginal.

---

## 4. Decode ceiling study

After the M26 route differential is understood and any evidence-backed current
route fix is applied, MIInfer should perform an explicit decode-ceiling study.

Required workloads:

- P512 steady decode
- P4K
- P8K
- P12K
- P16K

For each workload, attribute one complete token without overlapping spans.

At minimum record:

| Area | Required evidence |
|---|---|
| Recurrent projections | time, bytes, effective bandwidth |
| GDN/state work | time, state traffic, occupancy indicators |
| FFN Gate/Up/Down | time, bytes, effective bandwidth |
| Attention/QKV/O | time, KV traffic, context-dependent slope |
| Norm/conversion | time and intermediate traffic |
| LM head/argmax | time and bytes |
| Runtime/dispatch | launches, host critical path, GPU gaps |
| Transfers | H2D/D2H bytes and synchronization |
| Memory | allocations, active VRAM, duplicate representations |

Estimate:

```text
T_memory      = mandatory bytes / sustainable bandwidth
T_compute     = required arithmetic / sustainable compute
T_operator    ≈ max(T_memory, T_compute) + unavoidable local overhead
T_floor       = Σ required operator floors + unavoidable runtime overhead
headroom      = measured token time - plausible T_floor
```

The result must distinguish:

- physically required work
- contract-required work
- representation overhead
- runtime overhead
- implementation inefficiency
- unknown/unattributed cost

The study does not need to prove an absolute mathematical lower bound. It must
be good enough to reject impossible or low-value optimization targets.

---

## 5. Freeze rule

When a path is within a small measured distance of its plausible floor, freeze
that path.

Do not keep reopening it because a different tile, fusion, prefetch, or layout
might exist.

A frozen path can be reopened only when at least one of the following appears:

- new profiling evidence identifies material removable cost
- a new algorithm changes the modeled floor
- hardware/compiler behavior changes
- another feature changes the execution contract
- an external baseline demonstrates a materially better same-contract result

This rule is intended to prevent endless 0.x–2% tuning after the useful
optimization surface has been exhausted.

---

## 6. Compiler/runtime direction

A specialized compiler is **not** a roadmap objective by itself.

A compiler becomes justified only after MIInfer has repeatedly proven valuable
transformations such as:

- norm + quantization fusion
- paired projection execution
- model-specific quantized layouts
- recurrent-to-chunkwise reformulation
- device-resident decode state
- cache/state transformations
- hardware-specific scheduling rules

The required order is:

```text
discover transformation
  ↓
prove it manually
  ↓
repeat it successfully
  ↓
identify a reusable rule
  ↓
only then consider encoding it as a compiler/runtime transformation
```

Do not build an abstraction layer in the hope that optimization opportunities
will appear afterward.

---

## 7. Success metric

Raw PP and TG remain required qualification metrics, but they are not the only
product-level objective.

MIInfer should also measure the workload that originally exposed the serving
problem: real coding-agent interactions with multi-kilobyte prompts, tool
schemas, conversation history, prefix reuse, continuation, and rollback.

A useful final comparison includes:

- cold TTFT
- warm/prefix-reused TTFT
- steady decode
- end-to-end turn latency
- multi-turn wall clock
- context growth
- rollback/replay cost
- VRAM headroom

A runtime can be near the generic-runtime decode roofline and still have a
meaningful advantage on long-running agent workloads if it removes repeated
prefill/state work.

---

## 8. Research principle

The optimization question is no longer:

> What else can be made faster?

It is:

> What work is still removable, what is its quantified end-to-end value, and
> what evidence proves that the proposed implementation can remove it?

That question should gate every post-M26 performance milestone.
