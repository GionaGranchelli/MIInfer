# M31-0002T-0001 — First-divergence and weight-residency attribution

**Status: PARTIAL.** The run's configuration mismatch and the largest
source-derived memory contributor are identified. Exact logit causality,
full source tensor inventory, and safe recoverable bytes remain unresolved.
No GPU workload, profiling run, performance tuning, or production behavior
change was made.

## Baseline and provenance

- Base commit: `3c83447b1fede83d2e570f502671e02032c73e85`; the repository already
  had extensive dirty M31 source/docs/results. This report analyzes that
  working-tree overlay and preserves it.
- Host: Z840, MI50/gfx906, BDF `0000:06:00.0`; image manifest digest
  `sha256:bdc5ed42c985a6a333083e023225f248825a6e1511f38b6b50fbc7d5ace3fe9e`.
- Model SHA-256:
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.
- Same direct token-ID prompt: 1,024 tokens, fingerprint `bce3932a8fc5d121`,
  context 1,280, 16 requested outputs. llama.cpp commit
  `73a43d1f69345aee8bb186ef4b3172cef892f2e5`.
- MIInfer measured executable hash is recorded, but the original report does
  not bind it to a complete source-tree SHA. Treat source attribution as
  evidence about the inspected tree, not proof of the exact binary contents.

## Correctness

### Required final fields

| Field | Result |
|---|---|
| Model identity | Same model file SHA in both recorded runs |
| Tokenizer parity | Not applicable to this test: both consume supplied IDs directly |
| Generation config parity | **FAIL**: llama is plain greedy; MIInfer default repetition penalty is 1.15 |
| First divergence | Output step 7 (1-based), one recorded call |
| First eight IDs | llama `220,248046,198,248045,846,198,248046,198`; MIInfer `220,248046,198,248045,846,198,2523,248046` |
| Top-candidate comparison | Not captured |
| Logit differences | Not captured |
| Classification | `CONFIGURATION_MISMATCH`; numerical/runtime cause unresolved |
| Root cause | Penalty mismatch is the direct plausible cause; no proof it changed the step-7 winner |
| Correctness fixes | None; do not change model kernels based on this evidence |

The repeated token selected by llama at step 7 (248046) had appeared in the
generated prefix at step 2 and is penalized by MIInfer before its greedy
argmax. First align sampling (`repetition_penalty=1.0`, presence/frequency
penalties 0), then capture raw and post-penalty top candidates for the same
forced prefix. Only if the aligned raw logits materially differ should the
investigation trace model execution.

M31-0004's residency fix has no identified causal route in this cold, reset,
non-reuse run. M31-0005's fix applies to eager suffix attention, not this
graph-decode path. Their GPU regressions remain unvalidated.

## Memory

| Required field | Result |
|---|---|
| Source weight bytes | GGUF file: 17,106,775,008 B (15.932 GiB); exact active-tensor sum unresolved |
| MIInfer weight allocations | 24,068,487,168 B (24.068 GB / 22.416 GiB) |
| MIInfer actual weight residency | Same bytes allocated for model lifetime by the inspected ledger; per-tensor device high-water not independently captured |
| MIInfer sampled peak VRAM | 26,000,453,632 B (26.000 GB); allocator `device_peak_bytes` is null |
| llama.cpp sampled peak VRAM | 17,241,190,400 B (17.241 GB) |
| Total sampled VRAM difference | 8,759,263,232 B |
| MIInfer vs llama GPU model-buffer delta | 7,977,392,128 B |
| Duplicate-layout candidate | 7,130,316,800 B fused Gate/Up Wave layout co-resident with MMQ Gate/Up |
| Fused-layout overhead | 713,031,680 B: 534,773,760 B explicit padding + 178,257,920 B extra tile metadata |
| Q6 output-head layout expansion | 69,529,600 B |
| Proven safely recoverable | **0 B** |
| Unexplained | 131,913,728 B net model-buffer residual; 781,871,104 B of total short-run VRAM difference beyond GPU-weight delta; separate MIInfer post-call model-ledger residual 960,813,056 B |

The largest concrete difference is a second persistent FFN representation
(6.641 GiB) serving distinct prefill/decode layouts. The allocation is real;
its avoidability is not. The llama.cpp comparison ledger is from a separate
model/context probe and its 1K run has no category breakdown, so residuals
must stay unclassified.

## Validation and blockers

- Focused host-only CTests: **7/7 passed** in a fresh HIP-disabled CMake build
  directory under `/tmp`.
- Shared prompt test: passed with `g++ -std=c++17 -Wall -Wextra -Werror`.
- Broad host-only build: incomplete; existing CLI target requires
  `hip/hip_fp16.h` even with HIP disabled. No full project gate is claimed.
- HIP build: not attempted in this pass; existing M31 notes document the
  installed ROCm 6.2 linker failure on missing `libxml2.so.2`.
- GPU/logit validation: not run. The recorded run contains only selected IDs,
  not logits or top-k margins. A new forced-prefix/logit diagnostic must first
  be built and the guard rechecked; then one <=8-output guarded call is enough.

## Ranked next work

1. **Correctness:** test-only forced-prefix/raw-logit trace with penalties
   aligned to the reference; compare MIInfer and llama at output step 7.
2. **Memory:** CPU GGUF tensor inventory plus per-device allocation/layout
   ledger; then determine whether MMQ/fused FFN forms need simultaneous
   residency. Do not remove either form yet.
3. **Follow-up boundary:** if raw logits differ, compare MIInfer graph versus
   eager decode on the identical forced prefix and then isolate the first
   layer/state boundary. If logits agree within the correctness contract,
   retain the sampler fix in the benchmark harness and add regression tests.

**Next recommended goal:** `M31-0002T-0002 — Corrected Greedy Logit Trace and
Allocation Inventory`. It should add optional bench-only forced-prefix/top-k
logging, explicitly neutralize MIInfer penalties for reference parity, run one
guarded <=8-token Z840 check, and finish the CPU tensor/allocation inventory.
Do not begin another benchmark matrix or memory/kernel optimization until that
evidence is complete.

Detailed independent reviews: `m31-0002t-0001-divergence.md`,
`m31-0002t-0001-numerical-correctness.md`,
`m31-0002t-0001-weight-ledger.md`, and
`m31-0002t-0001-reference-and-tests.md` in this directory.
