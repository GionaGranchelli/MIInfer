# M31-LC-0004 — GDN Layer-0 Projection Correctness Isolation

**Status: IN PROGRESS — independent layer-0 projection references are established for the exact 2K case; causal substitution is blocked by a repeatable Z840 HSA fault during model allocation.**

## Scope and captured inputs

This work uses the frozen M31-LC-0003 model, prompt, and forced history. The new CPU capture is from llama.cpp revision `91c631b21d6e5d09e9c6659efdf6baeef5a44ddb`; its five-decision logits remain byte-identical to the frozen CPU oracle (`260c5e349910ef75aeba4d4473d23bf24c0bd31de923137c86e060e42624dda2`).

At GDN layer 0, prompt position 0, the 5,120-element normalized input is bitwise identical between CPU and fused GPU (SHA-256 `0a5b4e54f37a2c0dea4c6f35ce36b8fadcda44e00bac30a8ab92e7f1adec8ec8`). Captured tensor metadata identifies `blk.0.ssm_beta.weight` as F32 `[5120,48]` and `blk.0.attn_qkv.weight` as Q6_K `[5120,10240]`, with 4,200 bytes per QKV output row. The raw weight blobs and hashes are retained beside the output vectors.

## Independent operation reference

`analyze_projections.py` decodes canonical GGUF Q6_K blocks directly in NumPy, without calling ggml, then computes FP64 dot products and rounds outputs to F32. Beta uses its raw F32 weights with the same FP64 dot and F32 output rounding. A synthetic Q6_K block self-check exercises low/high nibble and two-bit fields.

The independent beta and QKV output hashes exactly match the separately captured CPU scalar projection vectors. This validates the reference's weight interpretation and confirms it uses the same logical input. It does not make the CPU GEMM or either GPU route the mathematical ground truth.

| Projection | Fused GPU relative L2 vs independent reference | CPU graph relative L2 | Top 10 absolute-channel overlap |
| --- | ---: | ---: | ---: |
| beta | `5.98e-8` | `2.59e-7` | 10/10 for both |
| QKV | `0.002858` | `0.003935` | 10/10 for both |

These measurements do not establish acceptable tolerances. The source uses the same Q6_K MMQ QKV path regardless of the experimental MMQ Gate/Up switch, but an MMQ-only run has not yet been captured for this milestone.

## Causal replay attempt

Diagnostic substitution hooks for one F32 beta vector or one QKV vector compile in the pinned ROCm 7.2.1 container. Each hook validates the input file size and finite values, waits for the work stream, then copies exactly the position-0 output vector before the following operation. This preserves tensor shape, dtype, and all later token rows.

The first MMQ-only capture and two fused captures using the rebuilt runner exited with SIGSEGV during model allocation, before the prefill or substitution hook. An older checkpointed-call binary from the same host and pinned container failed at the same allocation stage. The kernel journal reports a general protection fault in libc with the HSA runtime on the stack. After each failure the MI50 was idle (0% use, 28 C junction) with no KFD processes. No causal result is claimed; rerunning the operation requires resolving this host/runtime failure.

## Evidence hygiene

The 37 loose exploratory files left by M31-LC-0003 are preserved in the chunked archive and inventory under `results/m31-lc-0004/`. The archive was reassembled in memory, all 37 member sizes and hashes matched the inventory, and only then were the loose duplicates removed. `llama-cpu-scalar2-optrace.cpu.scalar-z-0.f32` remains explicitly identified as invalid because 3,114 of its 6,144 values are non-finite.

## Remaining work

- Run fused and MMQ-only first-token operation captures and verify their QKV/beta outputs with identical inputs.
- Substitute the independent beta output and QKV output separately at position 0, preserving the existing tensor shape, F32 type, token-major stride, and recurrent state, then measure downstream GDN and full-model changes.
- Record trace hashes and statuses, verify no new secret findings, and leave the worktree clean.
- Classify the result as a demonstrated defect, numerically acceptable with parity blocked, or unresolved. No operation tolerance or correctness pass is claimed here.
