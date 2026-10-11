# M31-LC-0008: Full-Model Oracle Reconciliation

## Status

Hard decision: **QUANTIZATION-ONLY NUMERICAL-CONTRACT DIFFERENCE ESTABLISHED at the first material CPU/GPU divergence.** The GPU QKV projection matches its Q6_K × captured MxQ8 arithmetic reference; the GPU recurrent scan matches the CPU result when both receive identical scan inputs. The frozen exact-token gate still fails at decision 5, so this is not a model qualification or route-equivalence pass.

## Frozen run identity

- Model: `Qwen3.8-27B-Q4_K_M.gguf`, SHA-256 `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.
- Prompt: 2,048 tokens, SHA-256 `5acd37f00d9e670f4b4e06d46fa12349e1dee06e356e6e73a186d74672f5bb47`.
- CPU: llama.cpp `d1d3c3396aa13a5f239109a822666c4870490ad5`, 0.4.1-dev build 162; zero GPU layers, KQV/op offload disabled, context 2304, batch/ubatch 512, 48 threads, greedy argmax.
- CPU runner SHA-256 `164a68a501b38ded7e315f26f0dcb3c41aa736103bd290b0bd1537c47ef76331`; its layer-trace variant SHA-256 `35929d8e91a9e11ed45ef95304141c1429d5c1b35a8568a4dbd37535d434b929`.
- GPU source revision for the frozen LC-0007 candidate: `6c4dc6a1af2e27692eb6586ca4b5e9e2f340e917`. Fused route, HIP graph on, aligned MMQ selector off.
- GPU trace executable SHA-256 `aac2e671c5ed16068bf2379e17c8eff1ea48ccfd5f7cb40d4ff5a23165f18afa`. This is an existing diagnostic build; an exact source-to-binary manifest for this diagnostic executable is not yet established, so it is supplemental evidence pending provenance closure.
- Host/image/device: Z840, pinned `localhost/miinfer-dev:rocm-7.2.1`, image digest `sha256:bdc5ed42c985a6a333083e023225f248825a6e1511f38b6b50fbc7d5ace3fe9e`, MI50 `gfx906`.

## Decision 1

The direct CPU runner repeated with the same raw-logit SHA-256 `8a23b51338410a30745207545814f9f369d504b287e9ba15aa16f766eb36e56e`. Both GPU routes are bitwise equal at decision 1 and reproduce fused raw-logit SHA-256 `78a1e4032a9f3253e7d4bc28926abf9d4939a77a4b86dfd8b5e319eb0737e92f`.

CPU versus GPU decision-1 relative L2 is `0.0235865`, RMSE `0.0826595`, max absolute error `0.6007606`. Both choose token 220. CPU's top-1/top-2 margin is `1.04316`; GPU's is `1.14472`. Token 198 scores `2.043241` (CPU) and `1.963192` (GPU); token 271 scores `1.546451` and `1.501432`. Their ranking is unchanged at decision 1, so the first token choice is not marginal.

## Final prompt-position layer boundaries

The captured position is token 2047, the last token of the 2,048-token prompt. CPU capture stores the 64 layer inputs; GPU capture stores embedding, 64 layer outputs, and final RMS hidden state. The embedding / input to layer 0 is bitwise identical. The first measured difference is the output of GDN layer 0: relative L2 `0.0258917`, RMSE `0.00581145`, max absolute error `0.2492933`. Relative L2 is `0.02845` after layer 1, `0.04679` after layer 2, `0.04421` after layer 4, `0.04728` after layer 14, peaks at `0.11079` after layer 34, then is `0.05971` after layer 62. Only the embedding boundary is bitwise equal.

This localizes the first material boundary; it does not yet identify which GDN layer-0 operation causes it, and floating-point difference alone is not evidence of a defect.

## Decision 5 with common history

CPU and fused GPU share the first four generated IDs `[220,248046,198,248045]`, so the captured fifth-decision vectors use the same token history. CPU selects 198 at `14.587840`; GPU selects 271 at `14.833824`. CPU scores 271 at `14.197399`; GPU scores 198 at `14.185873`. Decision-5 relative L2 is `0.182344`, RMSE `0.634609`, max absolute error `3.425908`. This verifies the divergence under common history but does not establish its cause.

## Operation-level findings

CPU and GPU operation traces were captured at prompt token 2047. The GPU state storage reports `state_position=0` during each 512-token tile, so the trace selects tile call 3 / row 511 by prompt order. The normalized input is bitwise equal; beta and alpha projection outputs differ by less than `3e-7` relative L2. The QKV output differs by relative L2 `0.003781`.

The QKV weight is GGUF `Q6_K` (type 14, `[5120,10240]`, 4,200-byte output-row stride). The captured non-affine MxQ8 activation row was re-gathered from the kernel's block-major workspace. A dequantized Q6_K × captured MxQ8 reference matches GPU QKV at relative L2 `2.797e-7` (RMSE `6.253e-7`, max `1.335e-5`); CPU QKV differs from the same reference by `0.003783` (RMSE `0.008457`, max `0.039657`). GPU4 and corrected GPU5 QKV vectors have identical SHA-256 `acbb9e0b2b0aaad15a241f3d92b3b5c4d92363faac8d6b792999bd60b8745c44`. This supports expected activation-quantization arithmetic as the cause of the QKV difference, not a QKV implementation defect.

The same reference at token 511 of the first tile gives CPU relative L2 `0.004845`, GPU `2.999e-7`, and CPU/GPU `0.004837`, using the same bitwise-equal normalized input. Thus the expected Q6_K × MxQ8 difference is present before the first state split.

The incoming recurrent state at the final 512-token tile differs by relative L2 `0.03554` after mapping the CPU state layout to GPU layout. Source inspection confirms both captures are the state immediately before that tile's recurrent scan: llama.cpp calls the `state_predelta` callback from `build_rs` before `build_recurrent_attn`, and MIInfer captures `incoming_state.d_state` before launching its GDN scan.

A four-boundary recapture locates the first measured state split. CPU and GPU states are equal at token 0; at token 512 the relative L2 is `0.00346493` (RMSE `0.000794951`, max `0.0538564`), at token 1024 it is `0.0426513` (RMSE `0.0103721`, max `4.32725`), and at token 1536 it is `0.0354858` (RMSE `0.00937576`, max `3.98446`). Thus the earliest observed recurrent-state divergence is produced within the first 512 prompt tokens. CPU and GPU recaptures retained their frozen raw-logit hashes exactly.

To isolate scan arithmetic, MIInfer's GPU GDN primitive was run on the CPU's first-tile query, key, value, beta, and decay arrays with a zero initial state. Its state after 512 tokens matches the CPU's state at token 512 at relative L2 `2.171e-7` (RMSE `4.982e-8`, max `6.676e-6`). This rules out a material GDN scan defect for identical inputs.

At token 511 of the first tile, CPU/GPU beta differ by relative L2 `9.63e-8`, and GPU decay versus `exp(CPU gate)` differs by `4.63e-8`. The normalized query and key differ by `0.00289` and `0.00487`. A corrected, stride-aware CPU view capture confirms CPU V is bitwise identical to the V slice of CPU convolution output; GPU V differs from that CPU V by relative L2 `0.000613`. These upstream input deltas follow the QKV projection's expected activation-quantization difference.

One earlier diagnostic had reported a large CPU/GPU V mismatch. That CPU row capture treated a non-contiguous GGML view as contiguous. The stride-aware recapture invalidated that result; it is excluded from the classification.

Diagnostic provenance: the corrected CPU trace runner binary SHA-256 is `31fc18823db3eec31cba43a39d2e64e4abf82f322c74e116ad52bb25b8de3dac` (source `experiments/m31-lc-0008/llama_cpu_ops.cpp`, SHA-256 `e15030922fb372deaf23bd77ecfe0ffc416d2f88bd876a50cf4fc118b4689e46`). The all-tile MIInfer trace binary SHA-256 is `945f59ce5d4d3f77e8f991cb67949cd928071276290aad910f8eba1a809eb4db` (recurrent source SHA-256 `f571c57384b5cf430039fb96da5d58e124d6b98299b383b503547d216b308d05`). The same-input GDN harness binary SHA-256 is `47aef7292db09501b52a950644792ac56f1c595d52e033b09fc61a02807afe54` (source `experiments/m31-lc-0008/gdn_input_reference.cpp`, SHA-256 `a00ebff3039164731e94fcf76083ce83c03e1c14e358a33c6c9776d1f178453f`), built and run in the pinned ROCm image.

The CPU full-tile capture runner used for the first-tile identical-input scan has binary SHA-256 `5834329bfdd468ffb980e3d722a98e74f9410ea436ff28f9323849b58f02eca2` and source SHA-256 `4671b6711649ad0c7f01e9a7b141d30dfff2542afb1e430f6a7f6bbf3bf38626`.

The corrected pinned-container GPU5 run reproduced decision-1 raw-logit SHA-256 `78a1e4032a9f3253e7d4bc28926abf9d4939a77a4b86dfd8b5e319eb0737e92f`; the thermal guard exited cleanly, with peak sampled junction temperature 68 C. The instrumented executable SHA-256 is `308635d89fc23ee57033d0b73cdf63e11ee897145603a4b29a568a1c2375df59`.

## Hard decision and qualification gate

The supported classification is an expected Q6_K activation-quantization / accumulation difference at the first material prefill divergence, not a demonstrated MIInfer GDN defect or a reference identity mismatch. Under the common first-four-token history, CPU ranks token 198 over 271 by `0.390441`, while GPU ranks 271 over 198 by `0.647951`; the fifth token therefore differs. The exact-token gate remains failed and must stay in place until separately reviewed. Do not promote aligned MMQ or start the long-context ladder under this goal.

The exact Q6_K tensor bytes were extracted from the pinned model for this calculation (SHA-256 `65b4f6f841cf18b0cc48b1db52868597af45d1f74578f33b010c4cb8c573edce`). They remain a local, ignored artifact because they are a slice of the model weights. The calculator and output reference vector are committed; rerunning it requires that extracted tensor alongside the pinned model identity above.

Artifacts in [`../../results/m31-lc-0008/z840-prefill/`](../../results/m31-lc-0008/z840-prefill/) contain the frozen logits and layer captures. [`../../results/m31-lc-0008/z840-operations/`](../../results/m31-lc-0008/z840-operations/) contains operation tensors, quantized activation bytes, tile-boundary recurrent states, same-input scan data, reference vectors, and run metrics; raw thermal guard logs remain local.
