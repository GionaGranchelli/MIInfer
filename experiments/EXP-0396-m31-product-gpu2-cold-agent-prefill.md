# EXP-0396 — M31 product GPU2 cold agent prefill

**Status:** RETEST  
**Milestone:** M31-PRODUCT-0001  
**Date:** 2026-10-12  
**Baseline commit:** 3991b6bf40f810b63a100c46981e16964bb5434b  
**Candidate commit:** 3991b6bf40f810b63a100c46981e16964bb5434b

## Question

Can the second MI50 on Machinist run the same large-context Oh My Pi request at the user-set 180 W cap, and return a useful tool action?

## Environment

- Host: Machinist X99. Z840 was not used.
- Active GPU: GPU 2, MI50 gfx906, PCI BDF `0000:88:00.0`, UUID `0x2152414173497dd3`.
- Idle GPU: GPU 0, MI50 gfx906, PCI BDF `0000:04:00.0`, UUID `0x21678e17348c2f7`.
- Both MI50s reported a 180 W power cap before the run.
- Model: Qwen3.8-27B Q4_K_M, SHA-256 `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.
- Image: `localhost/miinfer-dev:rocm-7.2.1`, digest `sha256:447907c697fb2c39ce9657401e354a70753d5672ad955d4ceb12de976b91224a`.
- MIInfer source: `3991b6bf40f810b63a100c46981e16964bb5434b`.
- MIInfer binary SHA-256: `6fbcd70f2e1e9ddf47f1291b688776a1ecb5193f681a8e8921907284db409a32`.
- Server settings: 98,304-token capacity, Q8 KV cache, persistent session reuse enabled.
- Thermal guard: warning at 80°C, stop at 90°C.

## Benchmark

Oh My Pi used the `m31-product-gpu2` profile with the same two context attachments and first-turn task used in the earlier GPU0 run. MIInfer processed 73,099 prompt tokens. The attachments and tool definitions made this larger than an exact 65,536-token prefix, and the attached history was not a representative coding repository context.

The local OMP command ran in print mode. Its terminal log contained only `Working...`. The inference server completed the request with `finish_reason=tool_calls`, but the OMP client did not save a resumable session before the run cutoff. The generated tool action and its execution were not verified.

## Results

| Measure | GPU2 result |
| --- | ---: |
| Prompt tokens processed | 73,099 |
| Prefill | 2,298,230 ms |
| Prefill throughput | 31.8067 tokens/s |
| First decode token | 2,298,230 ms |
| TTFT wall | 2,298,433 ms |
| Decode | 18,939.1 ms |
| Decode throughput | 12.7778 tokens/s |
| Generated tokens | 243 |
| Total request time | 2,317,290 ms |
| Reused prefix tokens | 0 |
| Persistent session restored | No |
| HTTP errors | 0 |
| Junction temperature during the measured run | 30–66°C |

The run completed without a thermal warning. The guard file also contains 80–81°C warnings from an earlier concurrent attempt, before this sample. Telemetry samples during this run occasionally exceeded the requested 180 W cap, with a sampled maximum of 230 W; treat the power readings as approximate until the cap telemetry is reconciled.

## Interpretation

GPU2 stayed cooler than GPU0 had in the earlier run, but this run did not show a latency benefit. The earlier GPU0 run measured 73,066 prompt tokens, 53.8335 tokens/s prefill, and 22.394 seconds for a cached follow-up. Only one cold-prefill sample per GPU is available, so these values do not establish a hardware winner.

Neither a fresh OMP process nor a resumed OMP transcript demonstrated a fast follow-up. Two requests entered prefill and remained active for about three minutes before I stopped the GPU2 container, which exited with code 137. The resumed transcript ID was `01a131bf-1af4-734d-9113-f0da8b510ec9`. An exact no-session replay of the original print-mode request also entered prefill and stayed active for one minute before I stopped it. Disk checkpoint restore did not produce a fast response in these attempts. The campaign has not demonstrated a useful OMP tool action, a cached follow-up on GPU2, or the eight-turn task.

## Decision

**RETEST.** Keep GPU2 out of the product decision until an interactive OMP session completes a tool action and a follow-up in the same saved session. Before another long run, establish how OMP can continue the same in-memory MIInfer session. Record the exact processed context and the returned tool name. Do not treat the cooler junction temperature as a speed win.

## Evidence

- Machinist campaign logs: `/home/machinist/m31-product-campaign/20261012T163632Z/gpu2/`.
- Completed inference request: request ID 5 in the `m31-product-gpu2` container log.
- Temperature and power samples: `gpu2/telemetry.csv`.
- Local OMP print-mode capture: `/tmp/m31-product-gpu2-first-turn.log`.
