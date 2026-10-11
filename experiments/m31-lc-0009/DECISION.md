# M31-LC-0009 decision record

## Decision

**Recommendation: retain the exact-token production gate for now.** Quantized inference can in principle be qualified with operation-level arithmetic evidence, deterministic state/logit replay, and independent functional checks without requiring every CPU/GPU greedy trajectory to match. The existing evidence does not yet support replacing the current gate. No selector, default, production gate, or promotion status changed.

**Overall assessment: BLOCKED for adopting a non-exact-token production contract.** The proposed three-contract framework is defined in [`PROTOCOL.md`](PROTOCOL.md). Tokenizer identity is now verified against the actual GGUF with the frozen model SHA, but only 16 of 24 required GPU prompt/configuration cells have valid data; the six-prompt CPU/GPU run matrix has not been completed.

The pinned Qwen tokenizer has 248,077 defined IDs, all of which match the GGUF vocabulary. The GGUF adds 243 reserved `[PAD]` IDs at 248077–248319; the three corpus special tokens match their frozen IDs. See [`tokenizer-gguf-verification.json`](tokenizer-gguf-verification.json).

## Evidence-based assessment

| Contract | Current evidence | Assessment |
| --- | --- | --- |
| O — quantization-aware operation reference | LC-0008's captured Q6_K × MxQ8 projection agrees with its independent arithmetic reference at relative L2 `2.797e-7`; same-input GDN scan agrees at `2.171e-7`. These cover selected operations and inputs only. | **Partial pass**; no corpus-wide operation bounds. |
| D — repeat and graph/eager determinism | LC-0006 reports byte-identical same-binary eager/graph logits for the frozen 2K workload. LC-0008 repeated the CPU full-model reference exactly and reproduced the GPU decision-1 logit hash. The LC-0009 CPU oracle now repeats greedy IDs/logits twice and matched-history logits three times on all six prompts. All 16 valid GPU cells have three byte-identical repeats for teacher-forced logits, forced and normal-greedy IDs, and recurrent/KV snapshots. On coding-64, reasoning-512, and ordinary-512, eager and HIP-graph results are byte-identical within both fused and aligned-MMQ routes. | **Partial pass**; eight required GPU cells remain pending. |
| X — CPU/GPU and fused/aligned-MMQ quality | LC-0008 finds the CPU and fused GPU disagree at decision 5 under common first-four-token history. All 16 valid LC-0009 cells (all four GPU configurations on ordinary-64, coding-64, reasoning-512, and ordinary-512) agree with CPU top-1 at all five teacher-forced decisions, with overall relative L2 from `0.0167` to `0.2987`. On all four prompts, fused and aligned-MMQ produce the same normal greedy IDs across eager and HIP-graph modes; their teacher-forced logits differ by route. GPU greedy output first diverges from CPU at token 31 on ordinary-64, token 39 on coding-64, token 6 on reasoning-512, and token 52 on ordinary-512. This covers only four prompts and does not establish a numerical acceptance envelope or functional pass. The earlier QSUM-only attempt omitted the MMQ-only selector and remains invalid. LC-0007 restores fused token parity on its frozen 2K graph workload but does not resolve CPU comparison or qualify the route. | **Blocked** for a replacement gate. |

## LC-0009 execution evidence

The pinned CPU reference matrix is complete for all six prompts. For each prompt, two fresh 64-token greedy generations had byte-identical IDs and logits, and three fresh teacher-forced captures were byte-identical, finite, and exactly `4,966,400` bytes. The runs used llama.cpp `d1d3c3396aa13a5f239109a822666c4870490ad5`, 48 CPU threads, zero GPU offload, a 2,304-token context, and F16 K/V. Per-prompt identities and capture hashes are in [`cpu-capture-manifest.json`](cpu-capture-manifest.json); raw CPU captures are retained on Machinist under `/home/machinist/experiments/m31-lc-0009/cpu-captures`.

The six 64-token CPU outputs were decoded with the pinned tokenizer and all stop in the model's reasoning text before reaching their frozen final-answer checks. This leaves functional acceptance **not evaluable** at the registered 64-token output cap; it is not evidence of an inference defect. Do not mark Contract X functionally passed until the completion checks are evaluated with a pre-registered output allowance.

The pinned ROCm 7.2.1 image configured and built the LC-0009 runner and HIP smoke target on the Z840; the smoke passed with exactly one visible `gfx906`. The ordinary-64, coding-64, reasoning-512, and ordinary-512 prompts now each have valid fused eager, fused HIP-graph, aligned-MMQ eager, and aligned-MMQ HIP-graph cells. Every cell has three byte-identical repeats for teacher-forced logits, forced IDs, normal-greedy IDs, and recurrent/KV snapshots. Per-cell records and CPU comparisons are checked in for the first two prompts; reasoning-512 and ordinary-512 each have a repeat-audit file plus four per-cell records and comparison files. Raw cells remain on Lake under `/mnt/lc9-lake/m31-lc-0009-captures/`. CPU/GPU top-1 agrees at all five teacher-forced decisions for every valid cell; overall relative L2 is `0.0167`–`0.2987`. GPU greedy output first diverges from CPU at token 31 for ordinary-64 (30-token common prefix), token 39 for coding-64 (38-token common prefix), token 6 for reasoning-512 (five-token common prefix), and token 52 for ordinary-512 (51-token common prefix). Fused and aligned-MMQ greedy IDs match across all four modes on each prompt, although route logits are not byte-identical. Outputs still stop before the frozen final-answer checks, and the logits at the first greedy mismatch are not separately captured. The 64-token prompts have only pre/post thermal samples; the 512-token prompts have 15-second sampling but no peak-temperature conclusion. The reasoning-512 sampler stop marker was set externally after its four runs completed; that SSH command then exited with status 0. The attempted QSUM-only capture omitted the required MMQ-only selector and remains invalid audit data in [`gpu-mmq-qsum-only-ordinary64-invalid.json`](gpu-mmq-qsum-only-ordinary64-invalid.json) and [`gpu-mmq-qsum-only-ordinary64-invalid-comparison.json`](gpu-mmq-qsum-only-ordinary64-invalid-comparison.json). Earlier hashes are in [`gpu-paired-smoke.json`](gpu-paired-smoke.json), [`gpu-fused-eager-ordinary64.json`](gpu-fused-eager-ordinary64.json), [`gpu-fused-eager-ordinary64-comparison.json`](gpu-fused-eager-ordinary64-comparison.json), [`gpu-aligned-mmq-graph-ordinary64.json`](gpu-aligned-mmq-graph-ordinary64.json), [`gpu-aligned-mmq-graph-ordinary64-comparison.json`](gpu-aligned-mmq-graph-ordinary64-comparison.json), [`gpu-repeat-smoke.json`](gpu-repeat-smoke.json), [`gpu-smoke.json`](gpu-smoke.json), and [`gpu-smoke-comparison.json`](gpu-smoke-comparison.json).

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

1. The GPU runner now builds and runs in the pinned image. Sixteen of 24 GPU prompt/configuration cells are valid, covering all four configurations on ordinary-64, coding-64, reasoning-512, and ordinary-512. Run the remaining eight cells on coding-2048 and reasoning-2048 across both GPU routes and eager/HIP-graph modes, with three repeats, separate normal greedy outputs, matched CPU-forced logits, and state captures.
2. Establish or justify the cross-engine recurrent/KV state mapping. Evaluate the prewritten functional checks; the current 64-token CPU outputs end before their final answers.
3. Finish Contract O references beyond the selected LC-0008 calibrations, retain/hash captures, compute the full metric set, and replace the partial assessments with a requirement-by-requirement result.
4. Revisit the recommendation only after evidence is complete; any exact-token gate change requires explicit approval.
