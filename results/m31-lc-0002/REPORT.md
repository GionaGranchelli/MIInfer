# M31-LC-0002 — Classify the 2K MMQ/Fused Numerical Divergence

## Decision

`UNRESOLVED` (classification D). The single guarded, instrumented 2K pair reproduces the established output sequences and captures valid pre-sampling F32 logits on both routes. It shows that the MMQ/fused distributions start differing before decision 5 and that the disputed tokens reverse rank at decision 5. The evidence establishes a real route-dependent numerical divergence, but has no independent full-model reference at this prompt/state to decide which route is numerically correct.

The existing exact token-parity gate remains unchanged. MMQ-only remains experimental and unqualified; the default fused route remains unchanged.

## Required fields

```text
M31-LC-0002: COMPLETE
SOURCE_SHA: f79b45d056c10ebe0f55822f803e19ebbe923b18
REMOTE_SHA: 9ec450edad4f4e0ac7c3cce12306dacee45585d2 (initial report/evidence publish; metadata correction is in a later pushed commit)
GPU_RUNS: 1 guarded matched 2K pair (control then MMQ-only)

DECISION_5_FUSED_TOP1: token 271, logit 14.833824, probability 0.320826
DECISION_5_MMQ_TOP1: token 74455, logit 15.149837, probability 0.460441
DECISION_5_DISPUTED_LOGITS: fused 271=14.833824 / 74455=13.723714; MMQ 271=14.084690 / 74455=15.149837
FUSED_TOP2_MARGIN: 0.147802 (token 271 over token 248046)
MMQ_TOP2_MARGIN: 1.065147 (token 74455 over token 271)

TOP10_OVERLAP: 9/10 at decision 5
PROBABILITY_DIVERGENCE: JS 0.108586 nats; total variation 0.430946
LOGIT_ERROR: decision 5 ordinary cosine 0.997885, centered cosine 0.992796; centered MAE 0.189999; centered RMS 0.241506; centered max abs 1.402044

OPERATION_REFERENCE: reused earlier layer-0 reference PASS (cosine 0.999998; relative L2 0.001991; existing thresholds 0.995 / 0.02); not the decision-5 input and not a full-model reference
GRAPH_EAGER_CONSISTENCY: NOT_RUN; HIP Graph was active for both captures
STATE_CORRECTNESS: fresh reset state and identical first four generated tokens verified; internal KV/recurrent state not independently compared

CLASSIFICATION: D. UNRESOLVED
CORRECTNESS_CRITERION_PROPOSAL: retain exact cross-route token parity as the existing selection gate; separately require same-state full-model comparison against an independent reference, same-route replay/state consistency, and finite valid output before performance eligibility. Derive error bounds from independently validated operation-level quantization and propagation measurements before candidate runs; do not use a universal cosine threshold or tune a threshold to this result.

EXPERIMENTAL_ROUTE_STATUS: MMQ-only remains experimental and fails the existing 2K cross-route parity gate
PRODUCTION_ROUTE_STATUS: default fused route unchanged; no production correctness or acceptance-policy change
NEXT_SINGLE_ACTION: capture or construct an independent full-model reference for this exact 2K prompt/state, then compare the route logits under a predeclared numerical error budget
COMMITS: 7d2b2f5 capture instrumentation; f79b45d harness test and opt-in handling; 9ec450e report and raw evidence
PUSH_VERIFIED: yes; 9ec450edad4f4e0ac7c3cce12306dacee45585d2 was confirmed on origin, then the metadata correction was pushed
```

## Evidence and analysis

The raw capture records five consecutive pre-sampling vectors of 248,320 native little-endian F32 values per route (4,966,400 bytes each). Both records validate `used_hip_graph=true`, the requested sampler, the pinned image/model, and clean guard teardown. The new outputs exactly match the earlier 2K run on each route:

```text
control:  [220, 248046, 198, 248045, 271, 248046, 198, 248045, 198, 248045, 248046, 198, 248045, 198, 248045, 248046]
mmq_only: [220, 248046, 198, 248045, 74455, 198, 248068, 198, 760, 1156, 682, 3106, 1092, 7701, 310, 381]
```

| Decision | Exact F32 vectors | Top-10 overlap | Fused / MMQ winner | JS divergence (nats) |
|---|---:|---:|---|---:|
| 1 | equal | 10/10 | 220 / 220 | 0 |
| 2 | differ | 10/10 | 248046 / 248046 | 0.001542 |
| 3 | differ | 9/10 | 198 / 198 | 1.13e-9 |
| 4 | differ | 9/10 | 248045 / 248045 | 1.41e-6 |
| 5 | differ | 9/10 | 271 / 74455 | 0.108586 |

Using a prereported rank criterion for material distribution divergence (the winner or top-10 candidate set changes), the first material rank divergence is decision 3. Exact raw vectors first differ at decision 2. At decision 5, token 271 is fourth in MMQ-only and token 74455 is fourth in fused; their relative logit ordering flips. The fused winner’s 0.147802 top-two margin is smaller than the 0.241506 centered RMS route error. This supports sensitivity to numerical changes, but does not identify the correct route. The top-10 probabilities, both disputed-token probabilities, all five decisions, and error metrics are in `capture/logit-analysis.json`.

The sampled VRAM saving in this instrumented pair is 7,248,769,024 bytes. Both calls passed functional and thermal guards; peak junction was 68 °C. The run reuses the fixed 2K prompt and same-source paired harness. Instrumented timing is retained as evidence only and is not used to qualify performance.

## Limits

The reused operation reference covers layer 0 only; it does not test identical decision-5 layer inputs or certify full-model logits. No standalone operation replay was run because no decision-5 Gate/Up tensors or independent full-model reference are available in the capture, so a synthetic operation comparison would not resolve this classification. Graph-versus-eager and hidden-state equivalence remain unverified. No production path, M31 acceptance policy, or context-scaling ladder was changed.

## Evidence files

- `capture/benchmark.json`: paired run record, source identity, metrics and guard state.
- `capture/*-pair-1.logits.f32`: the two five-decision raw-logit captures.
- `capture/*-pair-1.metrics`, `*.guard.log`, `*.jsonl`: raw timing, guard and telemetry evidence.
- `capture/logit-analysis.json`: reproducible per-decision comparisons.
- `source-manifest.json`: commit, binary, model, image and input source-file hashes.
- `SHA256SUMS`: checksums for the committed evidence bundle.
