# EXP-0339 — Mx attention decode reuse P512 qualification screen

**Status:** KEEP opt-in; promotion pending long-context/decode checks  
**Milestone:** M25  
**Date:** 2026-09-12  
**Candidate:** `260ccb4`  
**Reference:** `mx-llama.cpp@2e9d29fe736969160f17476ec6f0a6298cee6966`

## Question

Does reusing the M25-H/I Mx attention O and FFN weights for decode preserve
P512 performance and same-process state correctness while removing the
duplicate resident M23 attention representation?

## Environment

- AMD Instinct MI50 / gfx906 / Wave64
- Qwen3.8-27B-Q4_K_M, SHA-256
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- context capacity `1024`, exact 512-token repeated-fox prompt
- release binary, `MIINFER_MX_Q8_BATCH=0`, `MIINFER_MX_MMV` unset
- explicit H/I vector plus
  `MIINFER_PREFILL_WIDE_MX_REPACKED_ATTN_DECODE=1`
- three interleaved fresh-process candidate/control pairs
- continuous 250 ms telemetry: 816 samples for A/B and 127 for the state
  check; all sampled clocks were SCLK/MCLK `1606/1000 MHz`

## Correctness

The candidate passed the real same-process state gate:

```text
same_process_p512_check=PASS first_token=13477(brown)
continuation_token=37550 first_prefill_ms=2420.53
continuation_ms=512.503 repeat_prefill_ms=2519.99
```

All six fresh P512 processes completed with finite output and 512 processed
prompt tokens.

## Results

| path | P512 samples (ms) | median ms | median tok/s | allocation |
| --- | --- | ---: | ---: | ---: |
| candidate | 2513.95, 2410.03, 2412.24 | **2412.24** | **212.25** | 18,472,649,044 B |
| control | 2487.01, 2529.58, 2438.43 | **2487.01** | **205.87** | 21,993,242,964 B |

The candidate is `74.77 ms` faster by median (`3.01%`) and removes exactly
`3,520,753,920 B` of tracked allocation. The maximum sampled VRAM use was
`22,944,153,600 B`; the maximum junction temperature was `59 C`.

## Interpretation

The candidate preserves P512 correctness and improves this interleaved screen
relative to the qualified H/I control. It is a useful provisional SOTA result,
not a stretch match: the pinned oracle remains `2317.872 ms` / `220.892 tok/s`.

## Decision

**KEEP as opt-in.** Do not add the selector to `m25_hi_qualified` or make it
default yet. The P512/state gate passed, but long-context and intended decode
qualification remain outstanding.

## Follow-up

Run long-context P512/TG checks and the combined
`MIINFER_MX_MMV=1` decode workload. Promote only if those checks retain finite
state, no hot weight uploads, and an acceptable VRAM/latency tradeoff.
