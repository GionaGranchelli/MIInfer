# M31-LC-0003 CPU reference protocol

## Scope and frozen inputs

Use only the exact 2,048 prompt IDs and model from M31-LC-0002. The model SHA-256 is `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`; the prompt-ID SHA-256 is `5acd37f00d9e670f4b4e06d46fa12349e1dee06e356e6e73a186d74672f5bb47`. The four forced prior outputs are `[220, 248046, 198, 248045]`.

The pinned reference is the clean, committed llama.cpp source at `91c631b21d6e5d09e9c6659efdf6baeef5a44ddb`. That revision contains the `qwen35` architecture, including its recurrent linear-attention and dense-attention graphs. The reference run must load the exact GGUF and report 64 layers, hidden size 5120, and vocabulary size 248320. The CPU build must disable CUDA, HIP, Vulkan, and all GPU offload.

## Replay contract

- Feed the stored integer prompt IDs directly; do not tokenize text or insert BOS/EOS IDs.
- Process the prompt in four 512-token `llama_decode` batches, matching MIInfer's 512-token prefill macrotiles.
- Read raw F32 logits after the final prompt token for decision 1.
- Decode each of the four recorded history tokens separately, reading raw F32 logits after each. This yields decisions 2–5 with exactly the same prior token history used by both GPU captures.
- Use model-provided RoPE and architecture metadata, context 2304, one sequence, F16 K/V cache, no sampler, no penalties, no graph, and 24 CPU threads.
- Run the complete CPU replay twice from fresh contexts and require bitwise-identical F32 captures before treating the reference as reproducible.

The GPU captures were greedy, not overridden. Their recorded decision histories agree exactly through the first four tokens, so their decisions 1–5 have the same history as the forced CPU replay.

## Comparisons

For every route, decision, and vocabulary item, retain the full F32 vector. Report:

- RMSE and maximum absolute logit error against CPU.
- Relative L2 error `||route-reference||2 / ||reference||2`.
- Elementwise relative absolute error only where `abs(reference_logit) >= 1e-3`; report the excluded fraction so near-zero denominators are explicit.
- Top-10 token overlap, route argmax, CPU argmax, and the rank of the CPU winner in each route.
- Exact first generated-token disagreement under the fixed history.

The `1e-3` elementwise denominator floor and metrics are fixed before examining results. They describe numerical differences; they do not replace the existing exact cross-route token-parity acceptance gate. Being closer to CPU logits alone is not a PASS. A PASS requires demonstrated full-model parity under the existing gate or a localized defect with a corrected implementation that passes it.

## Localization gate

Start with decision 2. If one route tracks the CPU reference and the other does not, capture matching inputs and outputs at the earliest layer boundary where their activation streams differ. Include GDN recurrent state and GQA attention outputs only as needed to identify the first divergence. Replay an MMQ-sensitive Gate/Up operation only after capturing its exact shared input; compare fused, MMQ-only, and a scalar/reference computation on that tensor. Do not infer a kernel defect from logits alone or compare different layer inputs.

If neither route is supported, first verify comparable model metadata, 512-token prompt segmentation, F16 KV behavior, and forced positions. If still unresolved, stop with the precise mismatch; do not advance to longer contexts, persistent-agent benchmarks, or performance experiments.
