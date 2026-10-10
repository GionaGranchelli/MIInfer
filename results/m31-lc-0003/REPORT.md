# M31-LC-0003 — Independent Numerical Oracle

**Status: UNRESOLVED — the CPU oracle is reproducible, but neither GPU route passes the existing exact-token gate. The earliest layer-0 numerical differences are localized; evidence does not establish an incorrect kernel or justify a code fix.**

## Frozen replay and reproducibility

The comparison reuses the M31-LC-0002 2,048 prompt IDs (SHA-256 `5acd37f00d9e670f4b4e06d46fa12349e1dee06e356e6e73a186d74672f5bb47`), exact model (SHA-256 `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`), and forced history `[220, 248046, 198, 248045]`. The frozen fused and MMQ-only GPU captures share that four-token prefix. CPU replay used llama.cpp revision `91c631b21d6e5d09e9c6659efdf6baeef5a44ddb`, 512-token prompt batches, context 2,304, one sequence, 24 CPU threads, F16 K/V, and zero GPU layers. It loaded `qwen35 27B Q4_K - Medium` with 64 runtime layers, hidden size 5,120, and vocabulary 248,320.

Two fresh CPU contexts produced byte-identical five-decision F32 captures (SHA-256 `260c5e349910ef75aeba4d4473d23bf24c0bd31de923137c86e060e42624dda2`). This makes the reference reproducible under the pinned CPU implementation. The CPU greedy sequence is `[220, 248046, 198, 248045, 198]`.

## Full-vocabulary comparison

For each GPU route and decision, `comparison.json` retains RMSE, maximum absolute error, relative L2, elementwise relative absolute error for CPU logits with magnitude at least `1e-3` (and the excluded fraction), top-10 overlap, both argmax IDs, and the rank of the CPU winner in the route. The analyzer validates capture lengths and hashes, the prompt and forced history, GPU capture validity, and CPU reproducibility before comparing.

| Decision | CPU top token | Fused top token | Fused relative L2 | Fused top-10 overlap | MMQ-only top token | MMQ relative L2 | MMQ top-10 overlap |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 220 | 220 | 0.02496 | 8/10 | 220 | 0.02496 | 8/10 |
| 2 | 248046 | 248046 | 0.11042 | 9/10 | 248046 | 0.11960 | 9/10 |
| 3 | 198 | 198 | 0.09016 | 9/10 | 198 | 0.09757 | 9/10 |
| 4 | 248045 | 248045 | 0.13137 | 8/10 | 248045 | 0.13868 | 8/10 |
| 5 | 198 | 271 | 0.18235 | 10/10 | 74455 | 0.17670 | 9/10 |

Both GPU routes' raw logit vectors first differ bitwise from the CPU reference at decision 1. Their argmaxes still match the CPU through decision 4. At decision 5, the CPU winner (198) ranks third in fused logits and fifth in MMQ-only logits; the routes select different tokens from the CPU. MMQ-only has a smaller relative L2 at decision 5, while fused is closer at decisions 2–4. Closeness alone does not establish correctness.

## Layer and prompt-position localization

The CPU and fused-GPU embedding outputs match bitwise for all 2,048 prompt positions (SHA-256 `96ce7a52936fc3a7b8a2203731eb9e205b0c4f56a55b6bf2d2d9868fe6cebc97`). The first layer boundary that differs is the output of GDN layer 0, which becomes layer 1's input. Comparing every prompt position shows that layer 0 already differs at position 0: RMSE `0.00556809`, maximum absolute error `0.21034622`, relative L2 `0.01941864`; all 2,048 output rows differ. This rules out divergence accumulating only from recurrent state over the prompt.

## Layer-0 operation replay

At prompt position 0, the initial GDN state and attention RMS-normalized input match the CPU bitwise. The first paired operation output that differs bitwise is the beta projection (relative L2 `2.50e-7`, max absolute error `1.67e-6`). An independent scalar projection, using the same captured input, GGUF weights dequantized to F32, FP64 dot products, and F32 output rounding, puts CPU beta at relative L2 `2.59e-7` and GPU beta at `5.98e-8` from the scalar result. Alpha shows the same small scale: CPU `3.04e-7`, GPU `1.06e-7`.

The first larger paired difference is the QKV projection: CPU versus GPU relative L2 `0.00409`. Against the scalar projection, CPU is `0.00394` away and GPU is `0.00286` away. This is consistent with differing activation quantization and accumulation paths for Q4_K projections. It does not prove either path correct, and being closer on this one projection does not establish full-model correctness. The fused and MMQ-only modes share this prefill projection; their experimental switch selects decode FFN weights, so this shared difference does not explain their later token disagreement.

The trace-enabled fused replay completed all five decisions and reproduced the frozen fused logits byte for byte (SHA-256 `78a1e4032a9f3253e7d4bc28926abf9d4939a77a4b86dfd8b5e319eb0737e92f`). Its first layer output also matches the earlier fused row trace at position 0; the CPU operation trace output matches its independently captured CPU row trace. The scalar CPU replay retained the same frozen CPU logits hash `260c5e349910ef75aeba4d4473d23bf24c0bd31de923137c86e060e42624dda2`.

## Finding and remaining work

Neither GPU route is numerically equivalent to the CPU reference under the existing exact-token gate: both disagree with the CPU token at decision 5, and fused versus MMQ-only first differ at decision 2. The shared CPU/GPU difference is localized from identical layer input through the first projection branches and into the QKV MMQ result. Current evidence points to numerical differences in projection paths, not a recurrent-state or prompt-embedding mismatch. It does not establish an incorrect computation.

No corrective kernel change is justified by this evidence. Keep fused as default and MMQ-only experimental. Close this bounded investigation as **UNRESOLVED**, with exact-token parity still failing; do not loosen that gate based on this result. No 4K/8K ladder, performance study, or optimization work was performed.
