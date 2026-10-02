# Post-M26 Performance Roadmap

This document preserves the historical M26–M28 performance decision tree.
Its M26-C evidence remains valid, but old “next” statements are not current:
v0.2.0 is released, and the canonical post-release order is v0.2.x dogfood →
M29 → M30 → M31 → V3 dual-MI50 → second-model/generalisation. See
[`roadmap.md`](roadmap.md) for the authoritative scope and stage gates. M26-E
was not started and is not the current next milestone.

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

## M29 — Persistent Context Architecture

M29 is not merely a 64K–128K context benchmark. Its mission is a
placement-independent persistent logical context substrate, production 128K
qualification on one MI50, and semantics reusable if physical execution later
expands to two MI50s. Logical capacity may target 256K where inexpensive; that
does not require 256K physical backing on one card. `ContextSpace != HIP-VMM`:
HIP-VMM is an optional physical-backing experiment, and M29 continues with an
alternative if it fails. Preserve optimized kernel-facing physical views and
do not mandate per-access software page-table lookup. `N=1` remains first
class; the M29 two-shard/one-MI50 experiment is not dual-GPU performance
evidence. See the canonical roadmap and D032 for ownership, identity, N=1/N=2,
and the >1% decode-regression gate.

Planned stages are M29.0 long-context baseline (16K/32K/64K/128K), M29.0A
current gfx906 external-reference calibration, M29.1 backing feasibility, M29.2
ContextSpace/logical pages, M29.3 DeviceKvPool and DeviceKvShard, M29.4 two
logical shards on one MI50, M29.5 optimized physical view/attention integration,
and M29.6 production 128K qualification. Each stage requires its own focused
goal; this document authorizes no implementation.

M29.0A is deliberately bounded. Re-run the exact MIInfer Qwen3.8-27B-Q4_K_M
artifact against the ROCm 7.2.1 llama.cpp path from
`kyuz0/mi50-gfx906-toolboxes@a708a279` under the same qualified MI50 power,
clock, token-input, and measurement contract used by V2-0045. If equivalent
cells remain within 3% of the pinned upstream medians, preserve the refreshed
environment as a reproducibility control and stop. A reproducible >3% movement
requires attribution before MIInfer adopts any environment, library, kernel, or
runtime change. The toolbox's patched vLLM/Triton/FlashAttention stack is a
separate throughput/compatibility reference, not a direct single-stream decode
oracle. See [`gfx906-toolbox-reference.md`](gfx906-toolbox-reference.md).

---

## M30 — Agent Runtime Advantage

M29 defines where/how context exists. M30 defines session sharing, exact-prefix
and recurrent-state reuse, reference counting, COW, snapshots, fork, rollback,
append-only suffix prefill, and Tail-Replay over that substrate. Its primary
KPI is total wall-clock time over realistic long-running coding-agent
sessions—not only PP512/TG128—including roughly 20–50 turns, long prefixes,
tool calls/results, small suffixes, repeated continuation, and fork/rollback
where available.

---

## M31 — Single-MI50 Agent Frontier Qualification

M31 freezes and qualifies the completed N=1 generation before topology
changes. Qualification includes cold/warm TTFT; 8K/32K/64K/128K; prefill,
decode, VRAM, correctness, prefix reuse, suffix continuation, snapshot/fork and
rollback where available; coding-agent wall clock; and operational stability.
It answers how good the finished one-MI50 agent runtime is. Only after M31 does
the roadmap move to dual-GPU execution.

---

## V3 — Dual-MI50 Agent Engine

V3 targets two MI50s for one interactive coding-agent session, a Qwen 27B
Q4_K_M-class model, and 256K persistent physical context. `>=40 tok/s` decode
is a stretch target, not a promise. V3-0000 first measures the real PCIe/P2P,
bandwidth, latency, synchronization, IOMMU/ACS, VMM, and xGMI topology. GQA
head-parallel and GDN tensor-parallel execution remain hypotheses. Second-model
and generalisation work follows V3 in the current strategic order.

---

## Roadmap rule

A milestone number is not authorization to build.

Every post-M26 performance milestone begins with a measurable question and a
headroom estimate. If the evidence says the remaining optimization surface is
small, freeze that surface and move to the next research question.
