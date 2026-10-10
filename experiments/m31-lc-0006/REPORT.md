# M31-LC-0006: HIP Graph Execution Equivalence

## Status

Phase A instrumentation-free eager and graph runs are complete. Phase B traced the first mismatch to cold-prefill position metadata: eager used position 0 while graph used prompt position 2048. The position fix is pushed as `1b3a9cbd8fa4f72ec40a66601c0598d5524b7f9c` and rebuilt in the pinned OCI image. The first same-binary rerun is complete; exact computation equivalence remains unproven because post-fix raw logits still differ.

## Eager baseline

The fused control and MMQ Gate/Up route ran with the same binary, model, deterministic 2,048-token prompt, 8K profile, greedy sampler, and five generated tokens. `MIINFER_HIP_GRAPH=0` was recorded in both run records. Raw logits were captured for all five decisions (4,966,400 bytes per route).

| Route | Generated token IDs | Prefill (ms) | Decode (ms) | Peak sampled VRAM (bytes) |
|---|---|---:|---:|---:|
| Fused control | `[220, 16, 15, 15, 15]` | 8486.395 | 140.039 | 26,067,083,264 |
| MMQ Gate/Up | `[220, 16, 15, 15, 15]` | 8465.345 | 157.037 | 18,818,408,448 |

The two eager routes produced identical token IDs. This confirms the run pair is comparable at the token level; raw-logit equality has not yet been analyzed. One pair is descriptive evidence, not a statistical performance result.

## Graph comparison

The same workload and binary were run with `MIINFER_HIP_GRAPH=1`, with tracing disabled. The first decision's 248,320 raw logits were bitwise identical between eager and graph for each route, and both selected token `220`. At decision 2, after both modes consumed token `220`, every raw logit differed between eager and graph; both routes selected token `248046` in graph mode and token `16` in eager mode.

| Route | Eager IDs | Graph IDs | Decision 1 logits | Decision 2 logits |
|---|---|---|---|---|
| Fused control | `[220, 16, 15, 15, 15]` | `[220, 248046, 198, 248045, 271]` | Bitwise equal | All 248,320 values differ |
| MMQ Gate/Up | `[220, 16, 15, 15, 15]` | `[220, 248046, 198, 248045, 74455]` | Bitwise equal | All 248,320 values differ |

This localizes the first observable mismatch to the first decode after the shared token `220`. It does not yet identify which intermediate state or operation causes it. The graph run also enables graph capture for the configured suffix-prefill path; decision 1's exact logits match, but intermediate prefill state has not been compared.

The graph run records and raw captures are in [`results/m31-lc-0006/z840-untraced-graph/`](../../results/m31-lc-0006/z840-untraced-graph/). Each mode/route pair has one run, so performance figures remain descriptive only.

## Position-fix rerun

The fixed binary (`4963661cd7fd0025d09ebbd7eaa3a4782ef084539ee15b3638f271719a0874bd`) ran eager and graph modes on Z840 with the same 2,048-token prompt, model, and five-token decode. The fused control now produces the same token sequence in both modes. Its first-decision logits are bitwise identical; decisions 2–5 have no bitwise-equal values, with maximum absolute differences up to 1.114. The MMQ-only route matches through token four, but its fifth token differs (`271` eager, `74455` graph). Thus the position fix removes the control route's token mismatch in this sample, but does not establish identical decode computation.

| Route | Eager IDs | Graph IDs | Decision 1 logits | Remaining logits |
|---|---|---|---|---|
| Fused control | `[220, 248046, 198, 248045, 271]` | `[220, 248046, 198, 248045, 271]` | Bitwise equal | Decisions 2–5 all differ at every value |
| MMQ Gate/Up | `[220, 248046, 198, 248045, 271]` | `[220, 248046, 198, 248045, 74455]` | Bitwise equal | Decision 3 has one equal value; all other values differ, and decision 5 selects a different token |

The harness's separate route-selection gate returned `BLOCKED_OR_INCONCLUSIVE` for the untraced graph run; both two-configuration runs completed without a runtime or guard error. Full captures are in [`results/m31-lc-0006/z840-position-fix-eager/`](../../results/m31-lc-0006/z840-position-fix-eager/) and [`results/m31-lc-0006/z840-position-fix-graph/`](../../results/m31-lc-0006/z840-position-fix-graph/). Keep the exact-logit and exact-token gate; the graph implementation is not qualified yet.

Post-fix traces show recurrent layers 0, 1, and 2 are bitwise identical between eager and graph at position 2048. Each layer has 20 common fused-control tensors and 22 common MMQ tensors, including recurrent state before/after GDN, convolution history, QKV and Q/K/V convolution results, quantizer inputs, and layer output. Layer-zero captures are in [`results/m31-lc-0006/z840-position-fix-traced-eager/`](../../results/m31-lc-0006/z840-position-fix-traced-eager/) and [`results/m31-lc-0006/z840-position-fix-traced-graph/`](../../results/m31-lc-0006/z840-position-fix-traced-graph/); layer-one captures are in [`results/m31-lc-0006/z840-layer1-v2-traced-eager/`](../../results/m31-lc-0006/z840-layer1-v2-traced-eager/) and [`results/m31-lc-0006/z840-layer1-v2-traced-graph/`](../../results/m31-lc-0006/z840-layer1-v2-traced-graph/); layer-two captures are in [`results/m31-lc-0006/z840-layer2-v2-traced-eager/`](../../results/m31-lc-0006/z840-layer2-v2-traced-eager/) and [`results/m31-lc-0006/z840-layer2-v2-traced-graph/`](../../results/m31-lc-0006/z840-layer2-v2-traced-graph/).

## First GQA boundary comparison

On the same traced binary (source revision `1517e968a5bbc4aed959247ca1eb64c34397d3e0`, SHA-256 `2f2c9a793d73644d20ca59ba0e53103547716486e2d5f72cae2078d29504ba66`), the layer-2 output remains bitwise identical between eager and graph for both routes. At the first topology GQA block output (`block0.gqa3_output`, decode position 2048), the fused control is also bitwise identical (5,120/5,120 floats). The MMQ route differs at every float in this tensor, with maximum absolute difference `4.493379965e-4`. Thus the remaining divergence is downstream of layer 2 for both routes; for MMQ it is already observable at the first GQA output, while for fused control it occurs after that boundary.

The traced runs produce the same post-fix tokens and non-equivalent post-decision-1 logits seen in the instrumentation-free comparison. The harness summary returned `BLOCKED_OR_INCONCLUSIVE` for the graph captures, although each route call completed with valid tokens, clean guard exit, and no thermal abort. Full captures and run metadata are in [`results/m31-lc-0006/z840-gqa-traced-eager/`](../../results/m31-lc-0006/z840-gqa-traced-eager/) and [`results/m31-lc-0006/z840-gqa-traced-graph/`](../../results/m31-lc-0006/z840-gqa-traced-graph/); source provenance is in [`source-manifest-gqa-trace.json`](source-manifest-gqa-trace.json).

The instrumentation was checked against instrumentation-free eager and graph runs using this exact binary, source manifest, model, prompt, and route configuration. For both routes in both modes, all 1,241,600 raw logit floats (five decisions × 248,320 logits) are byte-identical with the layer-2/GQA trace enabled and with the layer-4/GQA trace enabled. Therefore these captures do not perturb the observed output in this bounded run. Untraced logits, records, and metrics are in [`results/m31-lc-0006/z840-final-untraced-eager/`](../../results/m31-lc-0006/z840-final-untraced-eager/) and [`results/m31-lc-0006/z840-final-untraced-graph/`](../../results/m31-lc-0006/z840-final-untraced-graph/).

To continue the fused-control trace, recurrent layer 4 (the next block's first GDN layer) was captured with the same binary. Its eager and graph inputs and outputs are bitwise identical. This moves the first fused-control divergence beyond layer 4; layers 0–2 and the preceding block's GQA output also match. In MMQ mode, layer-4 input already differs by the inherited GQA delta (`4.493379965e-4` maximum absolute), and layer-4 output differs by up to `9.202957153e-3`. The next useful fused-control boundary is recurrent layer 5 or 6. Selected tensors, per-route run records, and metrics are in [`results/m31-lc-0006/z840-layer4-traced-eager/`](../../results/m31-lc-0006/z840-layer4-traced-eager/) and [`results/m31-lc-0006/z840-layer4-traced-graph/`](../../results/m31-lc-0006/z840-layer4-traced-graph/).

Layer 5 was then captured. Fused control remains bitwise equal at the layer input and output (5,120 floats each), so its first remaining divergence is after layer 5. In MMQ mode, the inherited layer-4 input delta reaches `9.202957153e-3`; the layer-5 output delta is `2.078628540e-2`. Layer-5 tracing also left all four mode/route raw-logit arrays byte-identical to their same-binary untraced runs. Selected layer inputs/outputs, run records, and metrics are in [`results/m31-lc-0006/z840-layer5-traced-eager/`](../../results/m31-lc-0006/z840-layer5-traced-eager/) and [`results/m31-lc-0006/z840-layer5-traced-graph/`](../../results/m31-lc-0006/z840-layer5-traced-graph/).

Layer 6 also matches bitwise for fused control at its input and output (and at the traced QKV, convolution, and post-GDN state checkpoints). The first fused-control difference is therefore later than layer 6, at the next GQA block or a later operation. In MMQ mode, the inherited layer-5 difference is `2.078628540e-2` at layer-6 input; layer-6 output differs by up to `1.434922218e-2`. For all four mode/route cases, layer-6 tracing again left the raw logits byte-identical to the same-binary untraced runs. Selected layer input/output tensors and run records are in [`results/m31-lc-0006/z840-layer6-traced-eager/`](../../results/m31-lc-0006/z840-layer6-traced-eager/) and [`results/m31-lc-0006/z840-layer6-traced-graph/`](../../results/m31-lc-0006/z840-layer6-traced-graph/).

The next topology block's GQA output (block 1, model layer 7) was captured using a new graph-safe buffer. Fused control is still bitwise equal at both block 0 and block 1 GQA outputs (5,120 floats per output), so its first divergence lies later than layer 7. MMQ differs at block 1 GQA by up to `1.959152222e-1`; its block-0 GQA difference is `4.493379965e-4`. The extra block-1 graph copy has not yet been compared with an untraced run of this exact binary, so this boundary remains provisional until that check completes. Selected tensors, records, and metrics are in [`results/m31-lc-0006/z840-layer7-traced-eager/`](../../results/m31-lc-0006/z840-layer7-traced-eager/) and [`results/m31-lc-0006/z840-layer7-traced-graph/`](../../results/m31-lc-0006/z840-layer7-traced-graph/); source provenance is in [`source-manifest-block1-trace.json`](source-manifest-block1-trace.json).

## Phase B: eager trace validation

Built a trace-capable binary from source revision `c67fc073a8d508370d1e4423de71415504b4a9e9` inside the pinned OCI image (binary SHA-256 is in `source-manifest-graph-trace.json`). On this exact binary, the eager traced and untraced runs produced identical token IDs and byte-identical five-decision raw logits for both routes. This verifies that the eager diagnostic capture did not perturb the output.

The first eager recurrent trace is recorded as `decode-pos0`, while the graph decode state starts from prompt position 2048. Graph-safe capture completed for fused and MMQ routes. For both routes, layer input, recurrent matrix state, convolution history, attention normalization, beta, decay, Z, and QKV were bitwise identical between eager and graph before the convolution. The first differences were all Q/K/V convolution outputs; the recurrent matrix state and subsequent layer outputs differed after that.

This confirms the position mismatch as the cause. Eager decode calls the batch convolution with `state.position`; graph decode reads `DeviceDecodeState.position`. The cold-prefill path advances the device computation through the prompt but never stores the final `pos` in each host-side recurrent state. The reusable-prefix prefill path does store it. Thus the first eager decode uses position 0, skips the causal convolution history lags, and writes slot 0; graph uses position 2048, reads the correct prior slots, and also writes slot 0. Matching pre-decode state and QKV plus differing convolution outputs isolate the bug to this stale position metadata.

Eager trace and no-trace run evidence is in [`results/m31-lc-0006/z840-traced-eager/`](../../results/m31-lc-0006/z840-traced-eager/) and [`results/m31-lc-0006/z840-diagnostic-untraced-eager/`](../../results/m31-lc-0006/z840-diagnostic-untraced-eager/). Graph trace evidence is in [`results/m31-lc-0006/z840-traced-graph/`](../../results/m31-lc-0006/z840-traced-graph/).

## Provenance

- Source revision: `d4905bd5be1a4c3c32e466f319cee12cdd313f56`
- Binary SHA-256: `a71929408b3792db8c7a37dc2034fd6c4cce498d6c6a894849c426f13926c894`
- Model SHA-256: `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- GPU: Z840 MI50, BDF `0000:06:00.0`, unique ID `0x21678e17348c2f7`
- Container: `localhost/miinfer-dev:rocm-7.2.1`, digest recorded in `benchmark.json`
- `/dev/kfd` access worked in the rootless container. No privilege change was needed.

## Evidence

Complete runner output, per-route metrics and records, raw logits, prompt IDs, source manifest, environment snapshots, guard logs, and telemetry are in [`results/m31-lc-0006/z840-untraced-eager/`](../../results/m31-lc-0006/z840-untraced-eager/).
