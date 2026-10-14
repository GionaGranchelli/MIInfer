# EXP-0411: Q16 residual-query dot4 full-model check

## Decision

Reject this candidate for 64K execution. The 8K full-model run was faster, but
the final logits fail the existing Q8-control accuracy gate. Do not promote or
extend it to 64K.

## Setup

- Source: `codex/m31-prefill-q16-query` at `069351905ab1bb3c2e8d02987eaf71a8336ad8fe`.
- Runner: `miinfer-m31-fp16-full-model`, with the `--q8-only` and
  `--logits-out` benchmark switches in this report's commit, so control and
  candidate full-vocabulary outputs can be compared directly.
- Host: Machinist; pinned container `localhost/miinfer-dev:rocm-7.2.1`, digest
  `sha256:447907c697fb2c39ce9657401e354a70753d5672ad955d4ceb12de976b91224a`.
- Device: `ROCR_VISIBLE_DEVICES=0`; the HIP process reported
  `device_bdf=0000:04:00.0 architecture=gfx906:sramecc+:xnack-`.
- Model: `/home/machinist/models/Qwen3.8-27B-Q4_K_M.gguf`, SHA-256
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.
- Prompt: `/home/machinist/m31-perf-recovery-0001/build/mi50-release/m31-64k-context.txt`,
  SHA-256 `fa4a0de5db813f5f06b58de5b4b7df8f234c4c808e0965bd44c545e9087c462c`;
  first 8,192 tokenizer-produced tokens used in both runs.
- Capacity: 65,536 tokens. Thermal guard: 80 C warning / 85 C stop.

The host-side HIP probe verified that `ROCR_VISIBLE_DEVICES=0` exposes only
`0000:04:00.0`. Passing the BDF string itself as `ROCR_VISIBLE_DEVICES` exposes
no HIP device on this installed runtime.

## Commands

Both runs used the same built binary, model, prompt and capacity. The control
disabled dot4 and residual query; the candidate enabled both.

```sh
ROCR_VISIBLE_DEVICES=0 MIINFER_M31_Q8_DOT4_ATTENTION=0 \
MIINFER_M31_Q8_DOT4_QUERY_RESIDUAL=0 \
miinfer-m31-fp16-full-model MODEL.gguf --q8-only --tokens 8192 \
  --prompt-file m31-64k-context.txt --logits-out control-8k.f32

ROCR_VISIBLE_DEVICES=0 MIINFER_M31_Q8_DOT4_ATTENTION=1 \
MIINFER_M31_Q8_DOT4_QUERY_RESIDUAL=1 \
miinfer-m31-fp16-full-model MODEL.gguf --q8-only --tokens 8192 \
  --prompt-file m31-64k-context.txt --logits-out candidate-8k.f32
```

The actual runs used the pinned Podman image and the absolute mounted paths
recorded in the remote evidence directory
`/home/machinist/m31-prefill-0002-20261014/q16-query/`.

## Results

| Run | Prefill | Logits SHA-256 | Top-1 | Next token |
| --- | ---: | --- | ---: | ---: |
| Q8 production control | 54.663 s | `13d62d4b8f9a5869879e00ce875081a1df445ac460e18795f770abc9d78a431b` | 11 | 443 |
| Q8 dot4 + Q16 residual query | 36.171 s | `7c100f449de9d21bce0a30caedf0342dbd5a53fd3fb84809b4e3c08310920962` | 11 | 443 |

One sequential pair shows a 1.51x speedup (33.8% less prefill time). Both
outputs are finite and the first two generated tokens match. Full-logit
comparison does not pass:

```json
{"finite":true,"count":248320,"max_abs":1.0425357818603516,"cosine":0.9984842309051988,"top1_ref":11,"top1_candidate":11,"pass":false}
```

The required limits are `max_abs <= 0.01` and `cosine >= 0.9999`. The matching
top tokens do not compensate for this failure. Peak observed junction
temperature was 67 C; no thermal warning or stop occurred.

## Conclusion

The residual query plane did not recover full-model numerical agreement and
was worse than the earlier plain-dot4 8K comparison on both max absolute error
and cosine. Stop this Q16-residual variant. No 64K candidate run was made.
