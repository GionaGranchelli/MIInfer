# M31-LC-0008: Full-Model Oracle Reconciliation

## Status

In progress; correctness classification remains **UNRESOLVED**. The frozen 2K CPU and fused GPU runs are reproducible, and the first material layer boundary is now localized to the output of GDN layer 0. The operation-level CPU / quantization-aware reference / GPU comparison required to classify this difference is not complete.

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

## Blockers and next evidence

1. Capture GDN layer 0 operation tensors at final prompt position 2047 on both runtimes, with identical logical inputs.
2. Compare the identified operation against a quantization-aware reference using canonical Q4_K weights and the same input; decide whether the discrepancy is expected arithmetic or an implementation error.
3. Close exact source/binary provenance for the GPU diagnostic executable.

A fresh pinned-container build attempt stopped at CMake configuration because `find_package(hipblas CONFIG REQUIRED)` could not locate the hipblas package configuration. No repository source was modified for that failed attempt. The next diagnostic change should be limited to final-tile operation trace selection; no broad kernel investigation is authorized by this goal.

Artifacts in [`../../results/m31-lc-0008/z840-prefill/`](../../results/m31-lc-0008/z840-prefill/) contain CPU and GPU logits, layer captures, CPU run log, GPU metrics, and GPU thermal-guard log. Their SHA-256 values are recorded in the commit's file tree; the layer captures are native F32. GPU guard completed cleanly and peak junction temperature remained below 80 C.
