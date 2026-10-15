# EXP-0447 — GQA6 query-row reuse rejection

**Decision:** REJECTED — full-state exactness failed and the restored suffix was slower.

## Candidate

The experimental kernel kept the production 32-lane head-dimension mapping,
Q8 source values, XOR reduction and per-row sequential online-softmax update.
Each K/V vector was shared across four rows: two adjacent tokens by two query
heads. A CTA covered 16 tokens and two query heads. The route was opt-in under
`MIINFER_M31_Q8_GQA6_ROW_REUSE`; the production default was not changed.

The pinned ROCm 7.2.1 build completed. The gfx906 kernel used 124 VGPR and 47
SGPR, with zero spills and no LDS allocation.

## Restored-session screen

- Host/card: Machinist MI50, GPU2, BDF `0000:88:00.0`; selected with
  `ROCR_VISIBLE_DEVICES=2` and confirmed by the benchmark process.
- Model: Qwen3.8-27B Q4_K_M, SHA-256
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.
- Prompt SHA-256: `fa4a0de5db813f5f06b58de5b4b7df8f234c4c808e0965bd44c545e9087c462c`.
- Image: ROCm 7.2.1, `sha256:422b5d15d384d78cab724a0190ef739d31212674ebc1ae8f0f6cf538b1f5ea8f`.
- Workload: one interleaved direct-global control / row-reuse candidate pair;
  restored Q8/Q8 prefix of 73,099 tokens plus 1,024 tokenizer-backed suffix
  tokens, one HIP Graph replay tile.
- Both paths reported position 74,123 and dispatched the expected route on all
  16 layers with graph state present.

| Path | Restore | Suffix | First / second token |
| --- | ---: | ---: | --- |
| Direct-global control | 2.464 s | 20.894 s | 18369 / 198 |
| Row-reuse candidate | 2.526 s | 26.511 s | 18369 / 198 |

The candidate/control suffix ratio was `1.268` (0.788x speedup), a 26.8%
slowdown in this single screen. Logits were not exact and recurrent-state
traces were not exact; both paths produced equal first and second tokens and
all checked values were finite. The harness reported `all_correct=0` and exited
with status 3. It did not report the maximum logits or state error, so the
numerical divergence is not localized by this run.

Peak observed GPU2 junction temperature was 58 C, below the 80 C warning and
85 C stop thresholds. GPU0 ran a separate cold-prefill process; this pair used
GPU2 only.

## Disposition

Reject row reuse. Do not run the candidate on cold64K and do not promote it.
The implementation and its temporary pair-mode harness were reverted from the
working tree. The default remains the direct-global GQA6 path. The experiment
code remains available in branch history for diagnosis; raw run and telemetry
logs are at `/home/machinist/m31-gqa6-longctx-results/EXP-0452-gqa6-row-reuse-gpu2/gpu2/`.

The single timing pair is a rejection screen, not a qualified performance
distribution. The exactness failure alone blocks adoption.
