# EXP-0410 — V2-0043 next-K register prefetch

**Status: REJECTED for promotion.** The candidate preserved exact 64K logits,
but did not improve full-model cold prefill on the same MI50.

## Change

The opt-in `MIINFER_M31_V2_0043_K_PREFETCH=1` specialization loads the next
32-position FP16 K tile into four per-thread `uint4` values while the current
tile's V values are accumulated. It stages those registers into the existing
LDS layout on the next iteration. The KQ arithmetic, softmax, V accumulation,
split boundaries, and workspace remain unchanged. The benchmark runner uses
candidate value `16` to select this path. No default dispatch changed.

The code builds at commit `4fe0705df5b8b7d65b6f9836af32044c3c1ce585` with
`localhost/miinfer-dev:rocm-7.2.1`, image digest
`sha256:447907c697fb2c39ce9657401e354a70753d5672ad955d4ceb12de976b91224a`.

## Measurement contract

- Host: Machinist; model: Qwen3.8-27B-Q4_K_M, SHA-256
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.
- Prompt file SHA-256: `fa4a0de5db813f5f06b58de5b4b7df8f234c4c808e0965bd44c545e9087c462c`.
- GPU: `ROCR_VISIBLE_DEVICES=2`, BDF `0000:88:00.0`, gfx906; 180 W power cap.
- Prompt: first 65,536 of 198,637 tokenizer-produced tokens; configured
  capacity 69,632. The runner reports `prompt_tokens_processed=65536`.
- Thermal guard: 80 C warning / 85 C stop. Maximum sampled junction was
  62 C for control, 62 C for the first candidate run, and 63 C for the isolated
  repeat.

## Results

All full-logit outputs were finite and had the same SHA-256:
`acc091b9dbe66370599a67b1d427167272e256959218534d24a63c1ff83462fe`.

| Path | BDF | Processed tokens | Prefill time | Conditions |
| --- | --- | ---: | ---: | --- |
| Production control | `0000:88:00.0` | 65,536 | 525.802 s | Isolated |
| K-prefetch candidate | `0000:88:00.0` | 65,536 | 540.422 s | GPU0 ran an 8K workload early in this run |
| K-prefetch candidate repeat | `0000:88:00.0` | 65,536 | 540.551 s | Isolated |

The isolated candidate repeat is 14.749 s (2.80%) slower than the one isolated
control. The first candidate run took nearly the same time despite early
cross-GPU workload. This is a directional one-control comparison, not a
repeatability study; it provides no evidence of a speedup.

At 8K on GPU0, the single pair was 36.084 s control and 36.192 s candidate,
with identical logits SHA-256
`f21f2b349b9ec3ea73208d93fefac6cdc8f8e5261522c14469f03d6f10aac150`. Both
runs overlapped a GPU2 64K run, so this timing is only a correctness/smoke
result. A 2K candidate also matched the control hash exactly, but ran on a
different BDF and is not a speed comparison.

## Interpretation and next action

This register prefetch did not hide enough K-load latency to offset its added
live state and scheduling cost. Earlier MIInfer records also rejected
register-double-buffering on a related attention path (`EXP-V2-0020`) and found
no benefit from another register-prefetch variant (`EXP-0331`). Do not spend
more time tuning this prefetch.

The existing V2-0043 query/KV tiling already follows the pinned mx/upstream
`flash_attn_tile<256,256,16,2,false>` dataflow, as documented in
`EXP-V2-0043-reference-shaped-gqa-attention-rewrite.md`; this experiment copied
no external code. Strata's published workload is a 125B MoE model on newer
consumer GPUs, with MI50 listed only as experimental, so it does not provide a
comparable kernel result for this dense 27B workload.

Next, validate the unqualified Q8 dot4 attention route in a tokenizer-backed
full-model 8K A/B. Continue to 64K only if the full-model correctness gate
passes. Keep K-prefetch opt-in and disabled.
