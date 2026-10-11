# M31-LC-0009 decision record

## Decision

**Recommendation: retain the exact-token production gate for now.** Quantized inference can in principle be qualified with operation-level arithmetic evidence, deterministic state/logit replay, and independent functional checks without requiring every CPU/GPU greedy trajectory to match. The existing evidence does not yet support replacing the current gate. No selector, default, production gate, or promotion status changed.

**Overall assessment: BLOCKED for adopting a non-exact-token production contract.** The proposed three-contract framework is defined in [`PROTOCOL.md`](PROTOCOL.md). Its six-prompt run matrix has not been executed, and tokenizer identity has not yet been checked against the actual pinned GGUF metadata.

## Evidence-based assessment

| Contract | Current evidence | Assessment |
| --- | --- | --- |
| O — quantization-aware operation reference | LC-0008's captured Q6_K × MxQ8 projection agrees with its independent arithmetic reference at relative L2 `2.797e-7`; same-input GDN scan agrees at `2.171e-7`. These cover selected operations and inputs only. | **Partial pass**; no corpus-wide operation bounds. |
| D — repeat and graph/eager determinism | LC-0006 reports byte-identical same-binary eager/graph logits for the frozen 2K workload. LC-0008 repeated the CPU full-model reference exactly and reproduced the GPU decision-1 logit hash. | **Partial pass**; not three-run state/logit determinism over the new corpus. |
| X — CPU/GPU and fused/aligned-MMQ quality | LC-0008 finds the CPU and fused GPU disagree at decision 5 under common first-four-token history. The expected Q6_K activation-quantization difference is supported at the first material divergence, but no propagated full-model uncertainty envelope or multi-prompt functional result exists. LC-0007 aligned MMQ restores fused token parity on its frozen 2K graph workload, but does not resolve the CPU comparison. | **Blocked** for a replacement gate. |

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

1. Verify the official tokenizer revision and special-token mapping against the actual pinned GGUF on Z840.
2. Runner support for corpus prompt IDs, forced histories, and generated-token capture is implemented. The CPU oracle compiles and links against exact llama.cpp commit `d1d3c3396aa13a5f239109a822666c4870490ad5`; changed GPU sources pass HIP syntax checks. The metric analyzer smoke-check passes on existing LC-0003 captures, not LC-0009 data. A full build in the pinned ROCm 7.2.1 image and live capture checks remain pending.
3. Run the registered matrix, retain and hash raw captures, compute all listed metrics, and replace the partial assessments above with a requirement-by-requirement result.
4. Revisit the recommendation only after the evidence is complete; any gate change requires explicit approval.
