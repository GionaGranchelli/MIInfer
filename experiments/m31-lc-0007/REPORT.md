# M31-LC-0007: Q4_K Affine-Sum Contract Alignment

## Status

Graph-mode baseline reproduced before source edits. Layer-0 graph-safe Gate/Up capture is next. No LC-0007 source change has been made yet.

## Baseline reproduction

The base binary (`702618e756c2de5a2facc9420a6726b69e3d234054729e9eca3548089d10c2fb`) reproduced the LC-0006 graph-mode control and MMQ results on the Z840 with the same model, prompt, image, device, reset state, and five-token settings. Both guarded calls and raw captures validated.

| Route | Token IDs | Five-vector raw-logit SHA-256 | LC-0006 hash reproduced |
|---|---|---|---|
| Fused control | `[220, 248046, 198, 248045, 271]` | `78a1e4032a9f3253e7d4bc28926abf9d4939a77a4b86dfd8b5e319eb0737e92f` | Yes |
| MMQ Gate/Up | `[220, 248046, 198, 248045, 74455]` | `e53a0fe7b54629e437320107ca515c64060ee1ec387055dc3bf9e10719eeb1d4` | Yes |

The harness reports `BLOCKED_OR_INCONCLUSIVE` because fused/MMQ token parity is its route-promotion condition; these baseline route results are valid, and the divergence is the target of LC-0007. Full runner artifacts are in [`z840-baseline-graph/`](../../results/m31-lc-0007/z840-baseline-graph/), with a machine-readable reproduction check in [`baseline-reproduction.json`](baseline-reproduction.json).

## Contract detail to verify

The fused Gate/Up kernel computes its correction from the quantized integers and half-stored scale, multiplying the integer sum in FP32. The canonical `Q8_1Block::s` field separately stores a half-rounded `d * sum(q)`. LC-0007 will measure this distinction explicitly against the task's requested half-rounded selector and the independent projection reference; it will not assume the two forms are interchangeable.
