# Post-M26 Performance Roadmap

This document records the decision tree after M26-C route attribution closed.
EXP-0359 closes M26-C as `ROUTES_NOT_COMPARABLE`; M26-D was not opened. M26-E
on the qualified no-preset route is next.

See:

- [Performance Research Strategy](performance-research-strategy.md)
- [EXP-0358 — M26-C current decode-route differential](../experiments/EXP-0358-m26c-current-decode-route-differential.md)
- [EXP-0352 — M26-B historical decode-floor differential](../experiments/EXP-0352-m26b-historical-decode-floor-differential.md)

---

## M26-C — Close current route attribution

**Status: CLOSED — `ROUTES_NOT_COMPARABLE` (EXP-0359).**

### Goal

Measure the current legacy versus interactive decode-route differential from an
equivalent semantic state.

### Exit

Use the existing EXP-0358 stop gate:

- attribute at least 90% of the measured route delta, or
- show that no removable family accounts for at least 3 ms/token, or
- show that the remaining cost is required by the current contracts, or
- prove that an equivalent comparison is impossible without changing semantics

Do not infer a post-M26 optimization from historical M8/M9 timing.

Identical teacher-forced inputs expose route-state drift and a near-tie greedy
flip, but no accepted pairwise P512 tolerance establishes equivalence or a
correctness bug. No timing qualification was made. The qualified no-preset
route remains canonical; the interactive route is non-qualifying evidence, not
a performance target or baseline.

---

## M26-D — One evidence-backed floor reduction

**Status: NOT OPENED.** M26-C did not establish a valid same-contract
>=3 ms/token removable cost or a required correctness fix. Do not open M26-D
based on the interactive route's non-comparable timing.

### Authorization gate

Start M26-D only when M26-C identifies at least one of:

- >=3 ms/token of removable runtime/execution cost, or
- a correctness/contract issue that must be fixed before qualification

### Scope

Attack exactly the attributed cost.

Examples may include:

- synchronization
- route-specific state movement
- graph replay/update overhead
- avoidable transfers
- duplicated execution
- contract-specific dispatch overhead

Do not broaden M26-D into a general kernel campaign.

### Stop condition

Close M26-D when the identified cost has been removed, disproved, or reduced to
a level below the authorization threshold.

If M26-C finds no qualifying removable cost, record M26-D as **not opened**.

---

## M26-E — Decode context qualification

After the clean current route is established, measure the context curve under
one qualified execution contract.

Required points:

- P512
- P2K
- P4K
- P8K
- P12K
- P16K

Separate:

```text
fixed decode floor
+
context-dependent cost
```

Report:

| Measurement | Purpose |
|---|---|
| Wall ms/token and tok/s | Primary decode latency and throughput |
| GPU-event time/token | Separate measured GPU work from wall time |
| Context-dependent attention time | Measure the context scaling slope |
| Recurrent/GDN time | Check the expected near-constant cost |
| FFN and QKV/O time | Attribute the fixed floor and context contribution |
| LM-head time | Isolate fixed output cost |
| Kernel launches and synchronizations/token | Quantify runtime orchestration |
| H2D/D2H bytes and time/token | Quantify host involvement |
| Active KV bytes and recurrent-state bytes | Track context and fixed state footprint |
| VRAM use | Establish long-context feasibility |
| SCLK/HBM clocks, temperature, power and cap | Qualify hardware state |
| Correctness result | Confirm generated outputs and state remain valid |

M26-E must not invent a target from historical non-comparable results. Its job
is to establish the current qualified curve.

---

## M26-F — Decode ceiling study

This is a new mandatory decision milestone before a large decode-runtime
rewrite.

### Goal

Estimate the plausible physical and algorithmic floor of the qualified current
decode path and quantify how much optimization surface actually remains.

### Required evidence

For each major operator family and runtime component:

- measured time
- bytes moved
- effective bandwidth
- relevant arithmetic work
- launch count
- synchronization
- GPU gaps
- host critical-path time
- H2D/D2H traffic
- VRAM/state footprint
- plausible lower bound
- maximum removable end-to-end cost

Classify every material cost as:

- required model work
- required execution-contract work
- removable runtime work
- representation overhead
- implementation inefficiency
- unknown

### Exit artifact

Produce a table of the form:

| Component | Current cost | Plausible floor | Max removable | Evidence |
|---|---:|---:|---:|---|
| recurrent/GDN | TBD | TBD | TBD | profile/roofline |
| FFN | TBD | TBD | TBD | profile/roofline |
| attention | TBD | TBD | TBD | profile/context curve |
| norm/conversion | TBD | TBD | TBD | profile/traffic |
| LM head | TBD | TBD | TBD | profile/traffic |
| runtime/dispatch | TBD | TBD | TBD | host/GPU trace |
| transfers/state | TBD | TBD | TBD | counters/trace |

No performance claim should be made from a theoretical floor alone; the floor
is a research-prioritization tool.

---

## Decision gate after M26-F

```text
M26-C — CLOSED: ROUTES_NOT_COMPARABLE
        │
        ▼
M26-D — NOT OPENED
        │
        ▼
M26-E qualified context curve (P512/P2K/P4K/P8K/P12K/P16K)
        │
        ▼
M26-F decode ceiling study
        │
        ├── >=3 ms/token runtime headroom
        │       │
        │       ▼
        │      M27
        │  device-resident / unified decode work
        │
        ├── decode near plausible floor
        │       │
        │       ▼
        │   freeze decode
        │
        └─────────────────────────┐
                                  ▼
                                 M28
                         algorithmic prefill work
```

---

## M27 — Conditional unified/device-resident decode

M27 is no longer automatically authorized by the existence of runtime
complexity.

### Start gate

Start a substantial M27 performance rewrite only when M26-F identifies at
least **3 ms/token** of runtime/execution cost that the proposed architecture
can plausibly remove.

M27 may also proceed for an explicitly documented correctness or serving
requirement, but that rationale must be separate from the performance case.

### Candidate architecture

If authorized, M27 may investigate:

- device-resident decode state
- reusable graph/execution plans
- mapped observer/ring-buffer output
- removal of CPU synchronization from the token critical path
- one execution contract shared by benchmark, CLI, and server

### Exit

Require:

- correctness parity
- quantified removal of the targeted cost
- <=3% benchmark/CLI/server execution-contract divergence
- no new unexplained context-dependent penalty

If M26-F does not authorize M27, record it as deferred and freeze decode.

---

## M28 — Algorithmic prefill frontier

M28 is the main research milestone for structural gains once decode is either
qualified or frozen.

### Question

Can the hybrid model's recurrent/GDN prefill be evaluated with materially more
parallelism or less state traffic on gfx906?

Candidate directions include:

- chunkwise recurrence
- Split-WY-style formulations where mathematically valid
- GEMM-oriented contractions
- persistent/chunk-persistent execution
- prefill-specific state layout
- cross-operator scheduling

### Pre-implementation gate

Before a large implementation:

1. derive the arithmetic work
2. derive state/HBM traffic
3. estimate achievable GEMM/kernel geometry on gfx906
4. estimate whole-path speedup
5. reject the idea if the modeled end-to-end ceiling is below approximately 5%
6. prioritize ideas with clear double-digit headroom

M28 is not a request for another sequence of unbounded kernel experiments.

---

## M29 — Context architecture

Proceed only after decode and prefill contracts are stable enough to make
long-context evidence meaningful.

Focus:

- decoupled full-attention KV and recurrent state
- 64K–128K context behavior
- memory representation
- VRAM headroom
- zero-copy or low-copy state transitions

The existing 128K-with-headroom objective remains valid only if correctness and
performance remain qualified across the chosen execution contract.

---

## M30 — Agent runtime advantage

Measure the workload MIInfer ultimately needs to serve well.

Focus:

- exact prefix reuse
- recurrent-state reuse
- Tail-Replay / bounded suffix replay
- rollback
- multi-turn coding-agent traces
- cold versus warm TTFT
- end-to-end turn latency

The primary question is not only whether TG64 is faster. It is whether MIInfer
reduces total wall clock for real long-running agent interactions.

---

## M31 — Frontier qualification

Freeze implementation and refresh the comparison against the strongest
reproducible gfx906 baseline.

Qualification should include:

- PP across the selected prompt-size range
- TG across the selected context range
- memory usage
- correctness
- cold TTFT
- warm/prefix-reused TTFT
- multi-turn agent wall clock

Do not select a winner from one synthetic point. Preserve the full evidence
matrix.

---

## Roadmap rule

A milestone number is not authorization to build.

Every post-M26 performance milestone begins with a measurable question and a
headroom estimate. If the evidence says the remaining optimization surface is
small, freeze that surface and move to the next research question.
