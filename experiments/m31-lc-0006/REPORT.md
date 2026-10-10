# M31-LC-0006: HIP Graph Execution Equivalence

## Status

Phase A eager baseline is complete. Graph-mode comparison remains pending; no conclusion about HIP Graph equivalence is established yet.

## Eager baseline

The fused control and MMQ Gate/Up route ran with the same binary, model, deterministic 2,048-token prompt, 8K profile, greedy sampler, and five generated tokens. `MIINFER_HIP_GRAPH=0` was recorded in both run records. Raw logits were captured for all five decisions (4,966,400 bytes per route).

| Route | Generated token IDs | Prefill (ms) | Decode (ms) | Peak sampled VRAM (bytes) |
|---|---|---:|---:|---:|
| Fused control | `[220, 16, 15, 15, 15]` | 8486.395 | 140.039 | 26,067,083,264 |
| MMQ Gate/Up | `[220, 16, 15, 15, 15]` | 8465.345 | 157.037 | 18,818,408,448 |

The two eager routes produced identical token IDs. This confirms the run pair is comparable at the token level; raw-logit equality has not yet been analyzed. One pair is descriptive evidence, not a statistical performance result.

## Provenance

- Source revision: `d4905bd5be1a4c3c32e466f319cee12cdd313f56`
- Binary SHA-256: `a71929408b3792db8c7a37dc2034fd6c4cce498d6c6a894849c426f13926c894`
- Model SHA-256: `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- GPU: Z840 MI50, BDF `0000:06:00.0`, unique ID `0x21678e17348c2f7`
- Container: `localhost/miinfer-dev:rocm-7.2.1`, digest recorded in `benchmark.json`
- `/dev/kfd` access worked in the rootless container. No privilege change was needed.

## Evidence

Complete runner output, per-route metrics and records, raw logits, prompt IDs, source manifest, environment snapshots, guard logs, and telemetry are in [`results/m31-lc-0006/z840-untraced-eager/`](../../results/m31-lc-0006/z840-untraced-eager/).
