# MIInfer Performance Research Protocol

## Purpose

The purpose of this protocol is to **maximize useful MIInfer performance progress per experiment on gfx906 while minimizing speculative implementation churn**.

MIInfer previously spent too much effort testing plausible source-level optimizations before proving that their machine-level schedule was viable. Future performance work therefore follows an evidence-gated sequence:

```text
freeze target
  -> prove bottleneck
  -> study reference
  -> reconstruct machine schedule
  -> define one bounded hypothesis
  -> compiler/resource gate
  -> correctness gate
  -> isolated benchmark
  -> full-model Amdahl gate
  -> PROMOTE / REJECT / LEARN
```

This is process and evidence work. It does not authorize a new kernel or a new experiment by itself.

## One active performance frontier

Every performance area has exactly one state:

- **PRIMARY** — exactly one measurable optimization objective. This is the only area eligible for new performance implementation.
- **QUALIFIED** — meets the current required gate and is preserved; do not casually retune it.
- **DEFERRED** — potentially valuable, but not on the current critical path.
- **REJECTED** — an architectural family failed with sufficient evidence. Reopen it only when new evidence materially changes the premise; parameter tweaking is not new evidence.

The frozen frontier at protocol adoption is:

### PRIMARY

Qwen3.8-27B-Q4_K_M prefill performance on one MI50/gfx906. The next optimization target must be selected only after a refreshed full-model bottleneck attribution. No automatic EXP-0364 is authorized.

### QUALIFIED / PRESERVED

The current qualified decode path and M26-class decode performance, subject to their existing qualification records. Existing qualified paths remain controls while prefill attribution is refreshed.

### DEFERRED

128K context architecture; Tail-Replay/agent runtime; production server optimization; UI/installer; multi-GPU; and other non-critical performance features.

### REJECTED attention families

- [EXP-0360](../experiments/EXP-0360-gqa-tiled-prefill-candidate.md) — six-wave explicit GQA LDS sharing. It was correct but slower at every tested context; reuse benefit did not offset synchronization and resource cost.
- [EXP-0362](../experiments/EXP-0362-query-tiled-gqa2-prefill.md) — literal query-tiled BQ16/GQA2 geometry. It compiled into a spilled, much slower machine program; source geometry was not an execution schedule.
- [EXP-0363](../experiments/EXP-0363-spillfree-attention-state-schedule.md) — state-placement-only reconstruction. It removed spills and remained approximately 5.3x slower at P8K; resource placement alone did not reproduce the required KQ/V dataflow.

[EXP-0361](../experiments/EXP-0361-wave64-attention-architecture-analysis.md) is reference archaeology and evidence, not a rejected candidate.

## Mandatory pre-experiment gate

Before performance-sensitive code is modified, the experiment record or planning document must answer, with evidence where available:

```text
Objective:
Current full-model bottleneck:
Measured contribution to runtime:
Reference implementation:
Reference evidence:
Difference being tested:
Expected maximum end-to-end impact:
One independent variable:
Resource gate:
Correctness gate:
Performance gate:
Kill criteria:
Success criteria:
Action if rejected:
```

“No benchmark yet” is acceptable. “No idea what dominates” requires profiling or measurement first, not kernel code.

## Bottleneck-first and Amdahl gates

Do not begin an optimization merely because code looks inefficient, contains repeated loads, another architecture benefits from it, a paper recommends it, or a competitor appears to use a similar concept. First establish that the target is material to the current end-to-end workload.

Where possible, record full-model time, operator/phase time, dispatch count, context scaling, and the maximum possible Amdahl benefit. Estimate the end-to-end gain if the target operator became infinitely fast. If that ceiling is immaterial to the current goal, reject the target before implementation.

After every material isolated win, immediately measure the full workload and record the isolated improvement, full-model improvement, and new bottleneck distribution. After every material full-model improvement, refresh the bottleneck analysis. Do not continue optimizing an operator after its end-to-end ceiling becomes negligible.

## Reference-first and reproduce-before-invent

When a known reference is faster on the same or closely related hardware, study it before writing a candidate. For mx-llama.cpp on gfx906, record as applicable:

```text
runtime-selected kernel; grid and workgroup dimensions;
logical subgroup mapping; VGPR, SGPR, LDS, scratch and spills;
state placement and lifetime; data types and intermediate representations;
memory layout; phase decomposition; barriers; generated ISA
```

Labels such as “FlashAttention”, “tiling”, “fusion”, or “GQA reuse” are not sufficient. The execution/dataflow schedule is the relevant object.

The default order is:

```text
understand -> reconstruct -> reproduce -> measure -> improve
```

Any deliberate divergence from a proven reference must state why.

## Compiler-resource gate

For every new GPU performance kernel, collect before timing:

```text
VGPR, SGPR, VGPR spills, SGPR spills, private/scratch bytes,
LDS bytes, workgroup dimensions, wavefront size
```

Unexpected spilling is a hard stop unless intentional and justified. The normal gate is zero VGPR spills, zero SGPR spills, no unexpected private state, and LDS within expected architectural limits. Do not optimize for the smallest nominal VGPR count: 97 VGPR with zero spills can be healthier than 64 VGPR with 137 spills.

For material kernels, the effective implementation is HIP/C++ source plus compiler resource metadata plus generated gfx906 ISA. Inspect ISA when compiler behavior, scalarization, load/store behavior, instruction count, conversions, or resource usage is unexpected.

## One-variable rule

Change one architectural variable whenever practical. Keep geometry, data type, correctness contract, and unrelated layout choices fixed while testing state placement, for example. If multiple changes are inseparable, document why. Do not bundle new geometry, quantization, fusion, layout, and synchronization into one candidate.

## Benchmark ladder

Candidates that pass resource qualification proceed in this order:

```text
compiler/resource gate
  -> correctness
  -> isolated P512
  -> P2K
  -> P4K
  -> P8K
  -> full-model benchmark
  -> P16K+ only if justified by earlier scaling evidence
```

Do not spend long-context benchmark time on a candidate that has already failed earlier gates.

## Correctness gate

Record the exact or tolerance-based comparison, maximum error, tolerance source, and semantic/token consequences. Do not weaken a tolerance merely to qualify a candidate. Any numerical tradeoff needs a separate justification and documentation.

## Stop rules

If two materially different candidates from one architectural hypothesis fail, stop parameter exploration. Revisit the mental model or gather new evidence. A third candidate requires evidence that it tests a materially different premise.

Every experiment has exactly one primary disposition:

- **PROMOTE** — correctness, resource, isolated-performance, and applicable full-model gates pass.
- **REJECT** — the candidate fails its required gate and is not pursued.
- **LEARN** — the experiment produces durable architectural evidence but is not a production candidate.

A rejected experiment may also teach something; the repository disposition must still name one primary outcome.

## Rejected-family discipline

Do not retry a rejected family without recording:

```text
experiment
premise
evidence
why rejected
what new evidence would justify reopening it
```

Use the linked experiment record as the detailed source. Do not silently reopen a family through a renamed parameter sweep.

## Environment, thermal, and commit hygiene

The qualified MI50/gfx906 environment is part of the benchmark contract. Do not casually change ROCm/HIP compiler, clocks, memory clocks, power policy, benchmark flags, model file, quantization, or reference commit. Qualify required environment changes independently. Do not install profiling stacks that risk the ROCm environment unless the information is critical and no safer path exists.

The external workstation fan runs continuously at full speed; ROCm fan RPM/percentage telemetry is not authoritative. Use junction temperature, clock stability, power, reported throttle state where available, wall time, and energy where measurable. Do not infer fan failure from absent board telemetry.

Keep performance commits focused and bisectable. Preserve rejected experiment records. Keep generated graph/index refreshes in a separate commit where practical.

## Current next-frontier research plan

No performance implementation is authorized by this section. The missing evidence is a refreshed full-model prefill attribution on the current qualified MI50 setup, including phase/operator timing, dispatch count, context scaling, and an Amdahl ceiling against the current mx comparison. EXP-0360–0363 show that attention geometry/state-placement guesses are not sufficient; they do not prove attention remains the dominant current gap.

EXP-0364 completed the first attribution pass as `LEARN`, EXP-0365 confirmed
the partial-tail route defect, and EXP-0366 rejected the first opt-in bounded
contract. EXP-0367/0368 added a reusable state snapshot oracle and classified
the P640 L3 K/V delta as expected batched numerical drift: qualified B512
versus scalar P512 showed substantially larger K/V error envelopes. EXP-0369
corrected the P1664 comparison indexing: the first divergence is at the actual
partial-tail boundary, position 1536, where the selector changes the route
from scalar `run()` to batched partial attention. The current frontier is
stage-level qualification of that route. EXP-0370 localized divergence before
cache storage, and EXP-0371 localized the first unique difference to L7 input
hidden before attention normalization. EXP-0372 found that L6 input already
differs, alongside earlier L5/L6 recurrent state/history differences, so the
L6→L7 handoff is not the origin. The next PRIMARY is refreshed full-model
prefill attribution, followed only if justified by L5→L6 contract
qualification. No KV-write fix, tail geometry, or performance kernel is
authorized until that upstream contract is qualified. EXP-0373 then qualified
the aligned B128 route at P640/P1664: both 16-token continuations matched,
state drift was smooth, and the route recovered 77.89%/60.20% of clean
prefill wall. The next PRIMARY is arbitrary-remainder scheduling with the
validated B128 route; do not continue L5→L6 forensics without contradictory
model-boundary evidence.
EXP-0374 implemented the bounded repeated-B128 remainder scheduler behind an
opt-in selector. Exact P768 passed with zero scalar work and identical final
hidden/16-token continuation, but source review found that authored B128 chunks
failed the full-layer-major dispatch gate. EXP-0375 corrected the explicit
B512/B128 dispatch contract and proved the authored routes at P640/P768/P896/
P1022. Corrected timing shows approximately 13–15 seconds per B128 chunk,
making repeated-B128 state/runtime attribution the PRIMARY. Sub-128 residual
work remains blocked.

EXP-0381 corrected the historical wrapper-lifecycle question only: its
admitted A/B/C cases did not enable the count-64 composition guards. EXP-0382
is now the PRIMARY for the actual composed-B64 route; its route proof shows
zero scalar work, but the first-token result diverges from control and the
direct two-token case stops before `step()`. Do not run B64 performance,
P1022 timing, or B4 qualification until that semantic contract is resolved.
