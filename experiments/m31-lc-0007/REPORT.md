# M31-LC-0007: Q4_K Affine-Sum Contract Alignment

## Status

Graph-mode baseline reproduced before source edits, followed by a paired graph-safe layer-0 Gate/Up capture on the same base binary. No LC-0007 source change has been made yet.

## Baseline reproduction

The base binary (`702618e756c2de5a2facc9420a6726b69e3d234054729e9eca3548089d10c2fb`) reproduced the LC-0006 graph-mode control and MMQ results on the Z840 with the same model, prompt, image, device, reset state, and five-token settings. Both guarded calls and raw captures validated.

| Route | Token IDs | Five-vector raw-logit SHA-256 | LC-0006 hash reproduced |
|---|---|---|---|
| Fused control | `[220, 248046, 198, 248045, 271]` | `78a1e4032a9f3253e7d4bc28926abf9d4939a77a4b86dfd8b5e319eb0737e92f` | Yes |
| MMQ Gate/Up | `[220, 248046, 198, 248045, 74455]` | `e53a0fe7b54629e437320107ca515c64060ee1ec387055dc3bf9e10719eeb1d4` | Yes |

The harness reports `BLOCKED_OR_INCONCLUSIVE` because fused/MMQ token parity is its route-promotion condition; these baseline route results are valid, and the divergence is the target of LC-0007. Full runner artifacts are in [`z840-baseline-graph/`](../../results/m31-lc-0007/z840-baseline-graph/), with a machine-readable reproduction check in [`baseline-reproduction.json`](baseline-reproduction.json).

## Graph-safe Gate/Up capture

The instrumented fused and MMQ graph runs reproduced those same five-vector raw-logit hashes exactly. Both guarded runs exited cleanly, captured five raw logit vectors, and used HIP Graph. The first four generated tokens are shared; the fifth remains different (`271` fused, `74455` MMQ).

Layer-0 `layer_input`, recurrent state before and after GDN, attention outputs, `post_normalized`, and Q8 integers/scales are bitwise identical between routes. The captured FFN activation, down result, and layer output differ; this locates the first observable graph divergence after Gate/Up. The quantized activation values `q*d` match exactly, while the 160 Q8 sum groups differ between the original-input sum and `d * sum(q)`.

Using the pinned model's canonical layer-0 Q4_K Gate/Up blocks from the prior committed trace, the MMQ GPU outputs match an independent original-input-sum projection reference (Gate RMSE `5.14e-9`, Up RMSE `4.84e-9`). Substituting the requested half-rounded `FP16(d * sum(q))` correction into that same reference differs from current MMQ output (Gate relative L2 `0.00792`, Up `0.00849`). This confirms the existing MMQ correction contract and means parity from the proposed selector remains unproven; the selector must still be run end-to-end.

The harness reports `model_vram_bytes` of `25,128,019,968` for fused and `17,997,703,168` for MMQ on these runs (about `6.64 GiB` lower for MMQ). This is the harness's model allocation accounting, not a physical-residency claim. Full trace artifacts are in [`z840-gateup-graph-trace/`](../../results/m31-lc-0007/z840-gateup-graph-trace/).

## Selector build

Added the default-off `MIINFER_EXPERIMENTAL_MMQ_GATEUP_QSUM=1` selector. It changes only the affine sum metadata written by the existing Q8 quantizer at the two recurrent MMQ Gate/Up call sites; Q codes, scales, layouts, fused execution, and all other MMQ projections keep their existing behavior. The selector rejects malformed values and rejects `MIINFER_MX_Q8_BATCH=1` rather than silently using unsupported batch semantics.

The updated benchmark harness can run original-sum or selected-sum MMQ with the same executable. The parser self-check passed, and `e2-0003-checkpointed-call` built successfully in the pinned ROCm 7.2.1 image. Binary and changed-source hashes are in [`source-manifest-gateup-sum.json`](source-manifest-gateup-sum.json). Same-binary graph A/B results are next.

## Contract detail to verify

The fused Gate/Up kernel computes its correction from the quantized integers and half-stored scale, multiplying the integer sum in FP32. The canonical `Q8_1Block::s` field separately stores a half-rounded `d * sum(q)`. LC-0007 will measure this distinction explicitly against the task's requested half-rounded selector and the independent projection reference; it will not assume the two forms are interchangeable.
