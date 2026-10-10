# M31-LC-0006: HIP Graph Execution Equivalence

## Status

Phase A instrumentation-free eager and graph runs are complete. The graph-enabled routes diverge from eager at the second decision, after consuming the same first generated token. Phase B decode-stage localization is pending.

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

## Next step

Use a graph-safe trace to compare the first decode under identical token/state, starting with the recurrent state and layer-0 inputs/outputs. Allocate trace buffers before capture and retrieve them only after replay synchronization. First verify that tracing does not change eager raw logits.

## Provenance

- Source revision: `d4905bd5be1a4c3c32e466f319cee12cdd313f56`
- Binary SHA-256: `a71929408b3792db8c7a37dc2034fd6c4cce498d6c6a894849c426f13926c894`
- Model SHA-256: `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- GPU: Z840 MI50, BDF `0000:06:00.0`, unique ID `0x21678e17348c2f7`
- Container: `localhost/miinfer-dev:rocm-7.2.1`, digest recorded in `benchmark.json`
- `/dev/kfd` access worked in the rootless container. No privilege change was needed.

## Evidence

Complete runner output, per-route metrics and records, raw logits, prompt IDs, source manifest, environment snapshots, guard logs, and telemetry are in [`results/m31-lc-0006/z840-untraced-eager/`](../../results/m31-lc-0006/z840-untraced-eager/).
