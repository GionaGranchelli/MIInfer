# M31-0002T-0001 — First-divergence investigation

Status: configuration mismatch confirmed; exact step-7 logit cause unresolved.
This is based on the recorded 2026-10-09 guarded run and read-only source
inspection. No GPU workload or code change was made for this investigation.

## Input and generation equivalence

- Both harnesses pass token IDs directly; neither tokenizes text, inserts BOS,
  nor applies a chat template. Both use `make_prompt(1024, 77)` and record the
  same fingerprint `bce3932a8fc5d121`.
- Both load model SHA-256
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.
- Both use a 1,024-token prompt, 1,280-token context, and request 16 outputs.
  MIInfer's `8K` label is the selected profile; active context capacity is
  1,280 tokens.
- The important generation setting is **not equivalent**. llama.cpp installs
  only `llama_sampler_init_greedy()`. MIInfer sets temperature 0, top-p 1,
  top-k 1, but leaves `GenerateOptions::repetition_penalty` at its default
  1.15. MIInfer applies that penalty before its zero-temperature argmax.
  Setting top-k to 1 does not neutralize it.

The generated IDs establish the mismatch is at output step 7 (1-based), not a
counting offset:

| Step | llama.cpp | MIInfer |
|---:|---:|---:|
| 1 | 220 | 220 |
| 2 | 248046 | 248046 |
| 3 | 198 | 198 |
| 4 | 248045 | 248045 |
| 5 | 846 | 846 |
| 6 | 198 | 198 |
| 7 | 248046 | 2523 |
| 8 | 198 | 248046 |

Token 248046, selected by llama.cpp at step 7, was already generated at
MIInfer step 2. The deterministic prompt's IDs are at most 151643, so this
repeat came from generation, not the prompt. The MIInfer penalty is therefore a
direct plausible cause of the step-7 selection change, but there are no saved
logits or candidate margins to prove it flipped the winning token.

## Classification and evidence boundary

Run-level classification: `CONFIGURATION_MISMATCH`. The recorded comparison is
not an apples-to-apples greedy parity test. It does **not** establish whether
the underlying unmodified logits agree, whether numerical differences are
small, or whether an implementation defect exists. No top-k values, logit
margins, max/mean errors, or forced-prefix logits were recorded. This was one
comparison, not a repeated reproduction.

Sources: `bench/m31_0002_prompt.hpp`; `bench/m31_0002_checkpointed_call.cpp`;
`bench/m31_0002t_llama_reference.cpp`;
`include/miinfer/prefill_v2/model.hpp` (`GenerateOptions` defaults);
`src/prefill_v2/model.cpp` (penalty and greedy sampler); recorded IDs in
`results-m31-0002t-reduced-20261009/llama-token-ids.txt` and
`miinfer-metrics.txt`; run identity in
`results-m31-0002t-reduced-report-20261009.md`.

## Smallest decisive follow-up

1. Explicitly set MIInfer `repetition_penalty=1.0`, presence/frequency penalty
   to 0, matching the reference sampler.
2. On the same model, prompt IDs, context, and Z840 MI50, force the same first
   six output IDs in both engines; capture pre-sampling logits for the next
   decision. Record top-5 IDs/logits, top-1/top-2 margin, and full-vocabulary
   max/mean absolute differences. Also record MIInfer raw and post-penalty
   logits.
3. If the raw logits still materially differ, compare MIInfer graph and eager
   logits at that forced prefix before tracing earlier layers.

No profiling, repetitions, or long-context call is justified for this check.
