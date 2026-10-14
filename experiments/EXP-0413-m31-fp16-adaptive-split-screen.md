# EXP-0413 — FP16 adaptive attention split screen

**Status:** REJECTED at the 8K correctness gate; no 64K candidate run

## Question

Can using more K splits as the causal context grows reduce the FP16 cached-GQA
attention time without changing the model's accepted logits?

The opt-in route chooses `clamp(ceil((base_position + token_count) / 2048),
3, 32)` splits. The workspace already supports 32. With the flag unset, the
production route remains at three splits. The same helper is used in the
profiled and ordinary forward paths.

## Frozen workload

- Source: `codex/m31-fp16-split16` at `e97109a`; split policy introduced in
  `d7b9195`.
- Built binary SHA-256:
  `3343d7a33e0fb4dbb0c3439aed8685e4767042e1325c3e15b9a717fb30dd356b`.
- Host/GPU: Machinist, MI50 BDF `0000:04:00.0`, UUID `0x21678e17348c2f7`,
  `gfx906:sramecc+:xnack-`.
- Image: `localhost/miinfer-dev:rocm-7.2.1`, digest
  `sha256:447907c697fb2c39ce9657401e354a70753d5672ad955d4ceb12de976b91224a`.
- Model SHA-256:
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.
- Prompt SHA-256:
  `fa4a0de5db813f5f06b58de5b4b7df8f234c4c808e0965bd44c545e9087c462c`.
- Screen: first 8,192 tokens from that prompt, capacity 65,536, FP16/FP16 KV.
- Numerical gate: finite logits, max absolute difference <= 0.01, cosine >=
  0.9999, and matching first two greedy tokens.

## Results

One same-BDF run per path:

| Path | Prefill | First/second token | Full-logit comparison to control |
| --- | ---: | --- | --- |
| Production, 3 splits | 35.995 s | `11 / 443` | reference |
| Adaptive splits | 35.986 s | `11 / 443` | max abs `0.7134819`; cosine `0.9991644`; **FAIL** |
| Existing FP32-accumulation route, 32 splits | 39.027 s | `11 / 443` | max abs `0.6439674`; cosine `0.9993817`; **FAIL** |

The adaptive run logged its selected values: 3 splits at context end 2,048 and
4 splits at context end 7,168. Its 0.009 s timing difference is one sample and
does not establish a speedup. The existing FP32 route was 8.4% slower than the
production control and also failed the logit gate.

Both alternatives produced finite logits and the same greedy tokens, but both
exceed the project's full-logit error limits. The 65,536-token candidate run
was therefore not attempted. No production default changed.

## Decision

Reject split-count changes for the current FP16 accumulation contract. More
splits change the online-softmax partition and reduction order enough to fail
the accepted logit gate, while the 8K screen shows no measurable performance
gain. Do not spend a 64K run on this candidate.

Next attention work must preserve the existing three-split arithmetic contract
and reduce work inside that route, or demonstrate a different correctness
contract before another split-count experiment.
