# MIInfer Performance Engineering Policy (M31-R0)

Status: **proposed for ratification** on branch `process/m31-r0-aar-gates`. This becomes operational project policy only when reviewed and merged to the active production branch.

Tracking: [M31-R0 #9](https://github.com/GionaGranchelli/MIInfer/issues/9). Experiments: [E1 #6](https://github.com/GionaGranchelli/MIInfer/issues/6), [E2 #7](https://github.com/GionaGranchelli/MIInfer/issues/7), [E3 #8](https://github.com/GionaGranchelli/MIInfer/issues/8).

## P-01 — Temporary scope freeze

Until E1, E2, and E3 close and the decision checkpoint is held, do not expand the engine with unrelated features or architectures. Continue focused correctness, tooling, build repair, safety and bounded measurement work. Freeze applies to promotion onto the default single-MI50 production path, not to exploration in isolated branches.

## P-02 — One falsifiable hypothesis per performance change

State the observed symptom, model and workload, evidence, likely mechanism, expected metric effect, minimal falsification test, rejection criteria and revert path **before changing the hot path**.

## P-03 — Qualification and promotion

- **L0 OFFLINE / COMPONENT**: review, CPU/reference tests, compilation and relevant component experiments. Does not prove GPU correctness or end-to-end speed.
- **L1 GPU CORRECTNESS**: bounded, guarded short target-hardware run, including numerical/state correctness and memory safety. Required before promoting a changed default execution path.
- **L2 REPRESENTATIVE PERFORMANCE**: same-host baseline vs candidate on an appropriate medium workload (typically 32K/64K where meaningful), with numerical validity, variance/noise, memory and device state. Required for claims of production performance improvement.
- **L3 LONG-CONTEXT / RELEASE**: relevant 128K and release/stability qualification for long-context changes and final release. **Not required for every ordinary PR.**

Mark `CODE_COMPLETE`, `GPU_VALIDATED`, and `PERFORMANCE_QUALIFIED` separately. Source calculations and mock tests cannot establish a hardware performance win. A failed experiment may close with documented falsification and revert.

## P-04 — Validation debt and isolation

Inventory each unvalidated change by source SHA, affected path, default-path exposure, risk, owner, missing checks and disposition: `QUALIFY`, `ISOLATE`, `REVERT`, `RETAIN_AS_EXPERIMENT` or `BLOCKED`.

If isolation is warranted, prefer an isolated experimental branch; otherwise use a narrowly scoped compile-time or runtime opt-in defaulting **off**, with tests that prove the production default does not select the candidate. Do **not** hide a confirmed correctness fix if that restores a known bug; instead classify, test and retain or explicitly revert with reasoning. A flag is not a substitute for a qualification record.

## P-05 — Comparison identity and telemetry

Bind each benchmark to host/physical GPU identity (PCI BDF/UUID where available), source SHA, binary/build flags, target architecture (`gfx906`), dependency and ROCm/rocBLAS/hipBLAS versions, container digest, model bytes/hash (plus conversion and tokenizer identity), prompt token IDs, exact inference settings and run status.

On Z840, capture supported raw ROCm-SMI pre/post telemetry and in-run samples: edge/junction sensor identity, SCLK/MCLK and DPM state where exposed, GPU clocks, power cap/power, and throttle conditions when supported. Treat missing fields as unknown, not proof of normal operation. Record settings for **both** engines; they need not have identical compiler flags to be comparable. Do not blindly lock clocks/power.

Exclude hashing and model setup from prefill/decode timer intervals. Separately report loading, prefill, decode, thermal aborts, low-confidence energy and correctness. No performance claim from an aborted or numerically invalid result.

## P-06 — Numeric-comparison contract

E1 establishes identical tokenizer/input and forced-prefix token IDs. Compare all steps through the first divergence using aligned logits, top-k and margins, and where feasible numerically stable full-distribution KL/JS and cosine comparisons. Full-distribution similarity does not prove semantic equivalence; neither does top-1 disagreement alone prove a defect.

## P-07 — Memory accounting

Separate weight storage representation, workspace/scratch/reservations, simultaneously live duplication/temporary residency, and measured runtime/allocator overhead. Host-pinned allocations count as VRAM only if actually charged to device memory. Compare like-for-like categories and GB/GiB; do not assign a generic ROCm overhead without evidence.

Reported 1K **total** footprint delta = 26.00 - 17.24 = **8.76 GB** (subject to matching measurement semantics). MIInfer's 24.07 GB weight allocation subtotal versus llama.cpp's 17.24 GB *total* is **not** a valid category match.

## P-08 — Three-experiment checkpoint

After E1/E2/E3 close (or are explicitly BLOCKED with evidence), hold a review recorded in #9. Decide one next engineering optimization from established correctness, byte accounting and trusted benchmark evidence. No implicit continuation into M31-0007+.

## Current repository/provenance caveat

At policy proposal time, the connected GitHub `main` was `3991b6bf40f810b63a100c46981e16964bb5434b`, behind recent M31 commits reported from local hosts. `docs/current-state.md` separately identifies the v0.2.0 qualified source `94fad71ee19f539ce2ec0c7e100ad97d031dbefa`, with qualification only for that release scope. Do **not** reset main or label the M31 local work fully qualified without reconciling their actual refs.

## Enforcement scope

The PR template and a structural CI check can require declarations and evidence links; they cannot establish GPU validity, performance, or scientific correctness. Maintainer review and branch-protection configuration remain necessary before those checks are binding.
