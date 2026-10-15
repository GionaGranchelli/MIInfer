# EXP-0451 — Direct-global GQA6 cold 64K screen on GPU0

**Status:** INCONCLUSIVE — execution and exactness gates passed, comparison was contaminated

## Question

Does the exact direct-global Q8 GQA6 route improve an actual 65,536-token cold prefill relative to packed-U128 GQA6 on the same MI50?

## Fixed inputs

- Host: Machinist; MI50 GPU0, BDF `0000:04:00.0`, `gfx906`.
- Source: `28e467f07704985ec02486c553da4300a63615d8`; pinned ROCm 7.2.1 image digest `sha256:422b5d15d384d78cab724a0190ef739d31212674ebc1ae8f0f6cf538b1f5ea8f`.
- Model: Qwen3.8-27B Q4_K_M, SHA-256 `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.
- Tokenizer-backed prompt: SHA-256 `fa4a0de5db813f5f06b58de5b4b7df8f234c4c808e0965bd44c545e9087c462c`.
- Both paths processed exactly 65,536 tokens with Q8 KV and 69,632-token capacity. Each path ran separately with thermal guards enabled.

## Screen result

| Path | Cold prefill | Throughput | Exit | Correctness evidence |
| --- | ---: | ---: | ---: | --- |
| Packed-U128 control | 730.607 s | 89.7 tok/s | 0 | 248,320 finite logits; SHA-256 `9a33d3f3b190126806dee63074189df0f83b78503eac9f2f80e881babb16f482`; first tokens `271, 248068` |
| Direct-global candidate | 685.429 s | 95.6 tok/s | 0 | 248,320 finite logits; same SHA and first tokens |

The raw difference is 45.178 s (6.18% less time; 1.066x). Both runs reported 1,008 route-selection trace entries, used the expected GQA6 path, and reached 65,536 processed prompt tokens. Peak junction temperatures were 77 C and 79 C, below the 80 C warning and 85 C stop thresholds.

## Contamination and decision

This pair is not a valid attribution of the timing difference. The packed-U128 control overlapped the GPU2 row-reuse session experiment for approximately its final three minutes. The direct-global candidate overlapped a four-job HIP compilation for llama.cpp for most of its run. These loads can alter host-side scheduling, memory traffic and GPU timing. Correct outputs establish that both paths completed; they do not remove the timing confound.

Do not claim a cold-prefill speedup from this screen and do not change the default. Repeat the two paths serially only after the llama.cpp build and all other GPU/CPU workloads have ended. Preserve this result as an inconclusive screen rather than discard it.

The 1,008 route traces are emitted at graph capture/route selection; they are not live tile-progress telemetry.

Raw artifacts are on Machinist at `/home/machinist/m31-gqa6-longctx-results/EXP-0451-direct-global-cold64k-GPU0/`.
