# EXP-0366 — Partial-tail prefill execution contract

## Question and hypothesis

Can a remainder after complete B512 full-layer-major chunks use existing batched operators plus only a genuinely unsupported residue, without fake padding or a new math kernel? For `R=510`, the tested bounded contract was `384 + 64 + 62`.

## EXP-0365 evidence and baseline

The qualified/default route is unchanged. EXP-0365 measured fixed-capacity `P1022 = 80,973.89 ms` versus a B512 control near `2.49 s`; the pathological excess was approximately `78.48 s` (`96.9%` of the excess over P512).

## Existing primitive support matrix

| Stage | Existing implementation | Valid counts/alignment | State/causal requirement | Partial behavior/evidence |
|---|---|---|---|---|
| Embedding/copy | Per-token embedding/staging | Per-token | absolute position | Per-token; source path |
| Normalization | Fused/batched preparation and finish | Caller-dependent; normal path is 4-aligned | normalized activation | Batched only after caller reaches batch path |
| Recurrent QKV/Z, convolution, GDN/transition, SSM output | `prefill_wide` family | `>=128`, `<=capacity`, divisible by 64; normal preparation is 4-aligned | recurrent/convolution history and `base_position` | Wide for eligible chunks; otherwise normal or `run`; guards in `prefill_wide_*` |
| Recurrent FFN gate/up, SwiGLU, down, residual, next norm | `prefill_wide` / `finish_prefill_batch` | Wide `>=128` and 64-aligned; finish is 4-aligned | recurrent continuation | Deferred batch or B4 chunks; otherwise `run` |
| Full-attention Q/K/V/gate preparation, Q/K norm, RoPE, KV write | `prepare_prefill_batch` plus attention finish | Default exactly 64 or 512; opt-in aligned `128..512` | active KV, causal bounds, absolute RoPE | Unsupported counts reject and caller uses `run`; guard in `FullAttentionLayer::prepare_prefill_batch` |
| Full-attention causal attention and O projection | `finish_prefill_attention` | Positive prepared count within capacity | active K/V and causal positions | Batched only after preparation |
| Full-attention FFN gate/up, SwiGLU, down, residual, next norm | `finish_prefill_batch` / batch4 fallback | 4-aligned; wide when ready | layer continuation | Deferred batch/B4 or `run` |
| Layer-major dispatcher | `prefill_layer_major` / `prefill_full_layer_major_chunk` | Complete B512 fast path; partial route is stage-dependent | `base_position`, state/cache offsets | Route selection does not prove batched work |

The relevant source contracts are `prefill_layer_major`, `prepare_prefill_batch`, `prefill_wide`, `finish_prefill_attention`, and `finish_prefill_batch` in `tools/miinfer_cli.cpp` and `tools/qwen35_gpu_pipeline.hpp`. The classifications are: A only for selected generic APIs, B for normal preparation/finish, C/D for the wide recurrent family, E for default full-attention preparation, F for the `layer.run` fallback, and G for rejected preparation requests. A `count` parameter is not treated as proof that the complete stage is batched.

## EXP-0365 slow-path execution map for a 510-token tail

After the complete B512 chunk, the existing dispatcher aligns the first remainder portion to 448. Recurrent layers can enter `prefill_wide` for that portion, but default full-attention preparation rejects 448. The attention layer therefore calls per-token `layer.run`, and its deferred attention finish is not reached. The subsequent 64-token portion is below the wide threshold and is handled through per-token `run` at the layer-major call site. The final 62 tokens are also per-token.

| Tail portion | Recurrent family | Full-attention family | Classification |
|---|---|---|---|
| 448 | `prefill_wide` | `run` because preparation rejects 448 | mixed; attention loses batching |
| 64 | `run` because wide threshold is not met | `run` because batched-attention gate requires wide count | per-token |
| 62 | `run` | `run` | per-token |

The opt-in candidate changes the first portion to the aligned experimental contract. Its diagnostic profile recorded two aligned partial chunks and 62 scalar tokens, but that decomposition counter is not an exhaustive per-token dispatch counter.

## Candidate contract

`MIINFER_EXP0366_PARTIAL_TAIL=1` preserves complete B512 chunks, decomposes the post-B512 remainder into aligned chunks, and leaves the smallest residue on the existing scalar route. The full-attention preparation guard is widened only under this selector. No fake tokens, new allocation, duplicate weights, decode change, layout change, or new math kernel is used. The default and qualified selectors do not enable it.

## Oracle and correctness/state matrix

The existing fallback is the semantic oracle. The current CLI has no arbitrary-tail export for recurrent state, convolution history, active K/V, final hidden/logits, or token IDs. The required matrix is therefore recorded as not run rather than inferred from process exit:

| Tails | Recurrent state | Conv/history | Active K/V | Hidden/logits | 8–16 token continuation | Status |
|---|---|---|---|---|---|---|
| 1,2,3,4,63,64,65,127,128,129 | not run | not run | not run | not run | not run | blocked by oracle instrumentation |
| 255,256,257,447,448,449,510,511 | not run | not run | not run | not run | P513/P1022 process completion only | incomplete |

P513 candidate and default continuation both exited successfully, but token identity and semantic state were not exposed, so this is not equivalence. No numerical tolerance was weakened. Base positions after 0, 1, and 3 complete B512 chunks were not state-compared; P1022 covers one complete chunk only as a timing/counter observation.

## Resource and allocation gate

The release build completed. P1022 allocation remained `22,463,033,748` bytes. No new workspace, model representation, or transfer path was added. Since no new GPU kernel was introduced, VGPR/SGPR qualification does not apply to this routing experiment.

## Performance results

| Case | Result |
|---|---:|
| P512 control, default | 2,514.36 ms |
| P513 candidate, opt-in | 3,020.31 ms |
| P513 oracle continuation, default | 2,900.13 ms |
| P1022 candidate, opt-in | 82,260.50 ms |
| P1022 EXP-0365 baseline | 80,973.89 ms |

P1023, a representative medium tail, approximately P2K, and approximately P4K candidate comparisons were not run: correctness was incomplete and the candidate had already failed the clean P1022 screen. This is an intentional stop, not a passing result. Profiled timing is not an A/B claim because instrumentation changes synchronization and timing behavior.

The clean P1022 candidate regressed by about 1.29 seconds. Recovered share of the EXP-0365 pathological excess is therefore not positive; the experiment did not demonstrate removal of the excess. The context curve is not smooth.

## Decision

**REJECT** — primary repository disposition. The candidate fails correctness and performance gates.

**LEARN** — narrative result. Route selection changed, but the required state contract, exhaustive per-token counter, and end-to-end recovery were not established.

## Next PRIMARY frontier

Add non-invasive exclusive stage timing and a focused state/continuation oracle for an aligned partial tail. Identify the exact remaining serial stage or single missing primitive/count contract. If an existing primitive cannot express the required state-preserving partial count, record that operator and stop; do not tune another tail geometry or broadly optimize “partial prefill”.
