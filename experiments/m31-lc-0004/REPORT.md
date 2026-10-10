# M31-LC-0004 — GDN Layer-0 Projection Correctness Isolation

**Disposition: UNRESOLVED.** Independent references, matched fused/MMQ captures, and one-at-a-time beta/QKV substitutions are complete for the frozen 2K case. The routes agree bitwise through the captured position-0 layer-0 outputs, while QKV differs from the independent dot reference by relative L2 `0.002858`. No justified operation tolerance exists to classify that difference as correct or defective.

## Frozen case and input identity

The model SHA-256 is `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`. The exact 2,048 prompt IDs match M31-LC-0002 (SHA-256 `5acd37f00d9e670f4b4e06d46fa12349e1dee06e356e6e73a186d74672f5bb47`; prompt fingerprint `a998afbe8b675816`). The prior four generated tokens are `[220, 248046, 198, 248045]`; all runs use the same 2,304-token capacity, 512-token prefill tiles, greedy sampler, and pinned `miinfer-dev:rocm-7.2.1` image (digest `sha256:bdc5ed42c985a6a333083e023225f248825a6e1511f38b6b50fbc7d5ace3fe9e`). The selected MI50 is Z840 BDF `0000:06:00.0`, unique ID `0x21678e17348c2f7`.

At GDN layer 0, prompt position 0, the 5,120-element normalized input is bitwise identical in the CPU, fused, and MMQ captures (SHA-256 `0a5b4e54f37a2c0dea4c6f35ce36b8fadcda44e00bac30a8ab92e7f1adec8ec8`). Captured GGUF metadata identifies `blk.0.ssm_beta.weight` as F32 `[5120,48]` and `blk.0.attn_qkv.weight` as Q6_K `[5120,10240]`, with 4,200 bytes per QKV output row. Raw weights, vectors, and hashes are retained under `results/m31-lc-0003/reference/`.

## Independent projection reference

`analyze_projections.py` decodes canonical GGUF Q6_K blocks directly in NumPy without ggml, computes FP64 dot products, and rounds outputs to F32. Beta uses its raw F32 weights with the same dot and rounding. A synthetic Q6_K block self-check exercises the packed low/high nibble and two-bit fields. Both independent output vectors match the separately captured CPU scalar projection vectors bitwise.

| Projection | GPU relative L2 vs independent reference | GPU max absolute error | GPU RMSE | Top-10 absolute-channel overlap |
| --- | ---: | ---: | ---: | ---: |
| beta | `5.977e-8` | `4.768e-7` | `1.225e-7` | 10/10 |
| QKV | `0.002858` | `0.03556` | `0.005818` | 10/10 |

The fused and MMQ outputs for both projections are bitwise identical at position 0. The route switch affects FFN Gate/Up; source inspection and capture confirm it does not change these QKV/beta results. The captures also show no route difference in any recorded position-0 layer-0 stage through `post_ffn`. The final greedy outputs nevertheless diverge at decision 5: fused `[220, 248046, 198, 248045, 271]`, MMQ `[220, 248046, 198, 248045, 74455]`.

## Causal substitutions

The rebuilt runner binary SHA-256 is `7aab66e79ed992388b2f9cd15589d8b602c1398c86403f1035c55d1d8229d00e`. Matched baselines and four guarded replays substituted beta and QKV independently in each route. Every replacement validates its F32 byte length and finite values, synchronizes the stream, and copies only the position-0 output vector; shape, dtype, later token rows, and recurrent state handling are preserved. `analyze_replay.py` verifies that each injected output exactly equals its independent reference, route inputs match the CPU capture, raw-logit argmaxes match emitted tokens, and each intervention is compared with its own route baseline.

- **Beta:** Both routes show only tiny changes through GDN (beta-sigmoid max absolute change `1.192e-7`; raw GDN output `7.451e-9`; final GDN output `2.235e-8`). Each route retains its baseline five token IDs.
- **QKV:** Both routes show the same position-0 changes: raw GDN output relative L2 `0.002443`, max absolute change `0.000313`; final GDN output relative L2 `0.001846`, max absolute change `0.005171`. Each route again retains its baseline five token IDs. On decision 5, the fused top-token margin changes `0.1478` to `0.0854`; MMQ changes `1.0651` to `0.5095`. The routes still choose different tokens.

The substitutions establish that the reference QKV vector changes downstream GDN values and logits, but neither it nor beta substitution explains the fused/MMQ token disagreement for this case. The QKV error is shared by both routes. That does not establish whether the shared difference violates the operation's numerical contract.

## Replay launch diagnosis

The first four replay attempts exited before prefill because SELinux was enforcing and audit records denied `{ map }` on `/dev/kfd` for `container_t`; the kernel logged a resulting libc/HSA SIGSEGV. The known-good guarded Z840 launch includes `--security-opt=label=disable`. Replaying with that existing launch setting succeeded for both route captures, both matched baselines, and all four substitutions. The device remained below the 80 C warning threshold and every guarded process group exited cleanly. The failures were container labeling denials, not evidence of a GPU or model-allocation defect.

## Evidence hygiene and remaining limit

The 37 loose M31-LC-0003 exploratory files were preserved in the chunked archive and inventory under `results/m31-lc-0004/`; all member sizes and hashes matched before loose duplicates were removed. `llama-cpu-scalar2-optrace.cpu.scalar-z-0.f32` is marked invalid because 3,114 of 6,144 values are non-finite. The report and analyzer retain checksums for captured artifacts.

The task permits an `UNRESOLVED` disposition when the numerical cause cannot be classified. That is the result here: the independent FP64 dot is a reproducible comparison reference, not an existing production tolerance. Without an independently justified error bound for Q6_K dequantization plus the GPU projection, the QKV relative L2 cannot be called acceptable or defective. No kernel change or correctness pass is claimed.
