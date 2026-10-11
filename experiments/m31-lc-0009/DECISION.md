# M31-LC-0009 decision record

## Decision

**Recommendation: retain the exact-token production gate for now.** Quantized inference can in principle be qualified with operation-level arithmetic evidence, deterministic state/logit replay, and independent functional checks without requiring every CPU/GPU greedy trajectory to match. The existing evidence does not yet support replacing the current gate. No selector, default, production gate, or promotion status changed.

**Overall assessment: BLOCKED for adopting a non-exact-token production contract.** The proposed three-contract framework is defined in [`PROTOCOL.md`](PROTOCOL.md). The six-prompt CPU/GPU matrix is complete: all 24 GPU prompt/configuration cells have valid three-repeat data. The evidence still does not support replacing the exact-token production gate.

The pinned Qwen tokenizer has 248,077 defined IDs, all of which match the GGUF vocabulary. The GGUF adds 243 reserved `[PAD]` IDs at 248077–248319; the three corpus special tokens match their frozen IDs. See [`tokenizer-gguf-verification.json`](tokenizer-gguf-verification.json).

## Evidence-based assessment

| Contract | Current evidence | Assessment |
| --- | --- | --- |
| O — quantization-aware operation reference | LC-0008's captured Q6_K × MxQ8 projection agrees with its independent arithmetic reference at relative L2 `2.797e-7`; same-input GDN scan agrees at `2.171e-7`. These cover selected operations and inputs only. | **Partial pass**; no corpus-wide operation bounds. |
| D — repeat and graph/eager determinism | LC-0006 reports byte-identical same-binary eager/graph logits for the frozen 2K workload. LC-0008 repeated the CPU full-model reference exactly and reproduced the GPU decision-1 logit hash. The LC-0009 CPU oracle repeats greedy IDs/logits twice and matched-history logits three times on all six prompts. All 24 GPU cells have three byte-identical repeats for teacher-forced logits, forced and normal-greedy IDs, and recurrent/KV snapshots. | **Pass for the registered repeat matrix**; full-model correctness and cross-engine state mapping remain outside this contract result. |
| X — CPU/GPU and fused/aligned-MMQ quality | LC-0008 finds the CPU and fused GPU disagree at decision 5 under common first-four-token history. All 24 LC-0009 cells (all four GPU configurations on all six prompts) agree with CPU top-1 at all five teacher-forced decisions, with overall relative L2 from `0.0167` to `0.3243`. GPU greedy output first diverges from CPU at token 31 on ordinary-64, token 39 on coding-64, token 6 on reasoning-512, token 52 on ordinary-512, token 59 on coding-2048, and token 21 on reasoning-2048. Fused and aligned-MMQ normal greedy IDs match within 64 tokens on five prompts, but first differ at token 64 on coding-2048. No justified whole-model error envelope exists, the registered 64-token output cap does not reach the functional checks, and CPU/GPU recurrent/KV state has no established canonical mapping. | **Blocked** for a replacement gate. |

## LC-0009 execution evidence

The pinned CPU reference matrix is complete for all six prompts. For each prompt, two fresh 64-token greedy generations had byte-identical IDs and logits, and three fresh teacher-forced captures were byte-identical, finite, and exactly `4,966,400` bytes. The runs used llama.cpp `d1d3c3396aa13a5f239109a822666c4870490ad5`, 48 CPU threads, zero GPU offload, a 2,304-token context, and F16 K/V. Per-prompt identities and capture hashes are in [`cpu-capture-manifest.json`](cpu-capture-manifest.json); raw CPU captures are retained on Machinist under `/home/machinist/experiments/m31-lc-0009/cpu-captures`.

The six 64-token CPU outputs were decoded with the pinned tokenizer and all stop in the model's reasoning text before reaching their frozen final-answer checks. This leaves functional acceptance **not evaluable** at the registered 64-token output cap; it is not evidence of an inference defect. Do not mark Contract X functionally passed until the completion checks are evaluated with a pre-registered output allowance.

The pinned ROCm 7.2.1 image configured and built the LC-0009 runner and HIP smoke target on the Z840; the smoke passed with exactly one visible `gfx906`. All six prompts now have valid fused eager, fused HIP-graph, aligned-MMQ eager, and aligned-MMQ HIP-graph cells. Every cell has three byte-identical repeats for teacher-forced logits, forced IDs, normal-greedy IDs, and recurrent/KV snapshots. All 24 matrix cells now have per-cell records and CPU comparisons. A Z840-side SHA-256 inventory confirms all 36 core raw files per cell (teacher-forced logits, forced and greedy IDs, and recurrent/KV snapshots plus state metadata at decisions 1, 2, and 5) are present on Lake and byte-identical across all three repeats; all hashes previously recorded in cell metadata match. See [`gpu-lake-capture-inventory.json`](gpu-lake-capture-inventory.json). The early fused HIP-graph ordinary-64 repeat capture was originally stored under `/home`; it is now preserved on Lake at `/mnt/lc9-lake/m31-lc-0009-captures/lc9-fused-hip-graph-ordinary64/legacy-multirun`, with the paired-run archive alongside it. CPU/GPU top-1 agrees at all five teacher-forced decisions for every cell; overall relative L2 is `0.0167`–`0.3243`. GPU greedy output first diverges from CPU at token 31 for ordinary-64 (30-token common prefix), token 39 for coding-64 (38-token common prefix), token 6 for reasoning-512 (five-token common prefix), token 52 for ordinary-512 (51-token common prefix), token 59 for coding-2048 (58-token common prefix), and token 21 for reasoning-2048 (20-token common prefix). Fused and aligned-MMQ greedy IDs match within the 64-token cap on five prompts; on coding-2048 they first differ at token 64. Route logits differ across the fused and aligned-MMQ paths. Outputs still stop before the frozen final-answer checks, and logits at the first CPU greedy mismatch are not separately captured. The 64-token prompts have only pre/post thermal samples, except fused HIP-graph ordinary-64 whose repeat capture has no retained thermal log; longer prompts have 15-second sampling but no peak-temperature conclusion. The reasoning-512 sampler stop marker was set externally after its four runs completed; that SSH command then exited with status 0. The attempted QSUM-only capture omitted the required MMQ-only selector and remains invalid audit data in [`gpu-mmq-qsum-only-ordinary64-invalid.json`](gpu-mmq-qsum-only-ordinary64-invalid.json) and [`gpu-mmq-qsum-only-ordinary64-invalid-comparison.json`](gpu-mmq-qsum-only-ordinary64-invalid-comparison.json). Earlier hashes are in [`gpu-paired-smoke.json`](gpu-paired-smoke.json), [`gpu-fused-eager-ordinary64.json`](gpu-fused-eager-ordinary64.json), [`gpu-fused-eager-ordinary64-comparison.json`](gpu-fused-eager-ordinary64-comparison.json), [`gpu-aligned-mmq-graph-ordinary64.json`](gpu-aligned-mmq-graph-ordinary64.json), [`gpu-aligned-mmq-graph-ordinary64-comparison.json`](gpu-aligned-mmq-graph-ordinary64-comparison.json), [`gpu-repeat-smoke.json`](gpu-repeat-smoke.json), [`gpu-smoke.json`](gpu-smoke.json), and [`gpu-smoke-comparison.json`](gpu-smoke-comparison.json). The previously smoke-only fused HIP-graph ordinary-64 cell now has a consolidated record in [`gpu-fused-graph-ordinary64.json`](gpu-fused-graph-ordinary64.json) and comparison in [`gpu-fused-graph-ordinary64-comparison.json`](gpu-fused-graph-ordinary64-comparison.json).

State snapshots are MIInfer-native recurrent and KV layouts. A canonical mapping to llama.cpp's state remains unestablished, so these snapshots currently support within-route repeat checks and diagnosis only.

LC-0008's common-history decision-5 logits have relative L2 `0.182344` and maximum absolute error `3.425908`; those observed values do not justify a tolerance. LC-0002 likewise warns that a universal cosine threshold or a tolerance tuned to the observed result is not an acceptance rule. M31-PERF-0001's timings are excluded from this decision.

## Prior milestone evidence ledger

- [LC-0001](../../results/m31-lc-0001/REPORT.md): exact fused/MMQ greedy parity failed at token 5; full-model logits were not compared.
- [LC-0002](../../results/m31-lc-0002/REPORT.md): full logits establish route divergence, but no independent full-model reference classified it.
- [LC-0003](../../results/m31-lc-0003/REPORT.md): pinned CPU oracle repeats bitwise; both GPU routes fail CPU token parity, with early operation differences localized but not classified as defects.
- [LC-0004](../m31-lc-0004/REPORT.md): layer-0 QKV projection remains unclassified because no justified numerical bound existed.
- [LC-0005](../m31-lc-0005/REPORT.md): eager route difference localized; graph-enabled causal comparison and an accepted tolerance remained unresolved.
- [LC-0006](../m31-lc-0006/REPORT.md): GQA path fix restores same-binary eager/graph logit equality for the frozen workload; route equivalence remains separate.
- [LC-0007](../m31-lc-0007/REPORT.md): aligned-MMQ restores fused token parity on the frozen 2K graph case, but CPU full-model divergence remains unresolved and selector stays default-off.
- [LC-0008](../m31-lc-0008/STATUS.md): Q6_K × MxQ8 arithmetic explains the first material difference in the inspected path; decision-5 CPU/GPU token mismatch remains and is not qualification.

## Consequences

- Keep the production exact-token gate unchanged until all three contracts pass on the frozen corpus and an explicit gate change is approved.
- Keep aligned MMQ experimental and default-off. Its 2K graph parity is useful route evidence, not full qualification.
- Do not start M31-LC-0010 context scaling yet.

## Work still required

1. The registered 24-cell GPU matrix is complete. No additional GPU prompt/configuration cells remain.
2. Establish or justify the cross-engine recurrent/KV state mapping. Evaluate the prewritten functional checks with an output allowance that reaches the frozen final answers.
3. Finish Contract O references beyond the selected LC-0008 calibrations, and report the requirement-by-requirement result with defensible error bounds.
4. Revisit the recommendation only after evidence is complete; any exact-token gate change requires explicit approval.

### QSUM decode handoff (2026-10-12)

The QSUM decode reference remains **OPEN**. The available Contract O capture is prefill-only with QSUM disabled; the QSUM-only ordinary-64 cell is invalid because it omitted the aligned-MMQ selector. Tracked evidence has no raw represented operands for a one-token decode scalar reference. The handoff note [`QSUM-DECODE-HANDOFF.md`](QSUM-DECODE-HANDOFF.md) records the inspected evidence, the preserved dirty Machinist checkout, and the required next capture. Do not treat this as a Contract O pass or close LC-0009.
