# EXP-0412 — FP16 4Q LDS reuse on cached GQA path

Date: 2026-10-14  
Branch: `codex/m31-fp16-4q-lds`  
Candidate route commit: `469f28d`

## Question

Does staging FP16 K/V once for four adjacent query tokens improve the actual cached FP16 GQA route used by full-model prefill?

## Setup

- Host: Machinist; MI50 BDF `0000:88:00.0`; `ROCR_VISIBLE_DEVICES=2`
- Image: `localhost/miinfer-dev:rocm-7.2.1`, digest `sha256:447907c697fb2c39ce9657401e354a70753d5672ad955d4ceb12de976b91224a`
- Model: `Qwen3.8-27B-Q4_K_M.gguf`
- Prompt: `m31-64k-context.txt`; 8,192 tokens processed, capacity 65,536
- Same built source, same GPU, control then candidate; temperature guard 80°C warning / 85°C stop
- Control sets `MIINFER_M31_FP16_4Q_LDS_REUSE=0`; candidate sets it to `1`

## Dispatch correction

The initial 64K attempt did not exercise the candidate. Cached FP16 GQA dispatches through `launch_qwen35_kq_fragment_reuse_attention_batch_f16`, while the first feature-flag implementation was only inside the generic Split-K launcher. The initial candidate run therefore followed the production route and is not candidate performance evidence.

Commit `469f28d` moved the opt-in selection to the cached FP16 GQA launcher and emits `miinfer_m31_fp16_4q_lds_dispatch=1` when that route is selected. The candidate kernel supports the production launch's three splits. The 8K candidate log contains the dispatch marker.

## Result

| Path | Prefill time | Logits SHA-256 | Peak junction |
|---|---:|---|---:|
| Production control | 39.026 s | `f21f2b349b9ec3ea73208d93fefac6cdc8f8e5261522c14469f03d6f10aac150` | 58°C |
| FP16 4Q LDS candidate | 43.433 s | `814ed0aa3b6c0e25ece2a4bc390ccfdf1ec78243eed9d168fc638f3b4b913e0a` | 58°C |

Candidate was 11.3% slower. Both runs processed 8,192 tokens and produced finite logits; first two token IDs matched (`11`, `443`). Full-logit comparison failed: `max_abs=0.5183954`, cosine `0.9996130` (gate: max abs <= 0.01 and cosine >= 0.9999).

## Decision

Reject this candidate for the cached FP16 GQA route. Do not run it at 64K. Keep the feature opt-in and disabled by default; its dispatch now targets the intended path, but the valid routed screen fails both correctness and speed.

The prior 64K run before the dispatch correction produced identical control/candidate logits at 526.065/526.589 s, but both executed the original GQA kernel. Those timings are explicitly excluded as candidate evidence.

## Evidence files

- `evidence/routed-gpu2-control-f16-8k.log`
- `evidence/routed-gpu2-control-f16-8k.temps`
- `evidence/routed-gpu2-control-f16-8k.f32`
- `evidence/routed-gpu2-candidate-f16-4q-8k.log`
- `evidence/routed-gpu2-candidate-f16-4q-8k.temps`
- `evidence/routed-gpu2-candidate-f16-4q-8k.f32`

These files remain in the untracked Machinist evidence directory at `/home/machinist/m31-fp16-4q-lane2-20261014/evidence/`; they were not added to Git.
