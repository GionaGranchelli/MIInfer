# EXP-0030 — M5-C10a refreshed P64 production profile

**Status:** CLOSED — MEASUREMENT-ONLY
**Milestone:** M5
**Date:** 2026-09-01
**Production policy:** shared Gate/Up Q8 activation reuse
**Model:** Qwen3-8B Q4_0 (`Qwen3-8B-q4_0-b968826d.gguf`)

## 1. Question

With M5-C9c shared Gate/Up Q8 reuse production-selected, where does P64 decode
time now go and what is the next highest-value avoidable cost?

## 2. Method

The existing position audit ran the trace-free MI50 decode path at positions
1, 8, 16, 32, and 64 with GPU argmax and shared reuse enabled. The clean wall
and lightweight whole-token HIP timings are separate from the intrusive
deferred per-operation attribution. The latter is used for ranking only.

Environment capture and JSON output are retained under
`bench/results/20260901T134000Z-m5c10a/`. The observed telemetry was
approximately 930 MHz SCLK and 350 MHz MCLK; absolute rates are therefore
low-clock qualified.

## 3. P64 result

| Metric | P64 |
|---|---:|
| Clean production wall | 20.213 ms |
| Whole-token GPU event | 20.121 ms |
| Deferred attributed GPU time | 27.828 ms |
| Dispatches | 1553 |
| Synchronization sites | 38 |
| Temporary allocations | 0 |
| Copy accounting | 589,828 bytes |

Detailed category attribution:

| Category | GPU ms | Dispatches |
|---|---:|---:|
| FFN projection | 6.963 | 108 |
| Attention | 4.935 | 36 |
| Normalization | 2.978 | 289 |
| LM head | 2.876 | 1 |
| Conversion | 2.278 | 324 |
| Q/K/V projection | 1.690 | 108 |
| Residual | 0.512 | 72 |
| RoPE | 0.567 | 72 |
| Argmax | 0.275 | 1 |
| KV store | 0.268 | 36 |
| Embedding | 0.039 | 1 |

Fine-grained FFN attribution:

| Stage | GPU ms | Dispatches |
|---|---:|---:|
| FFN normalization | 0.915 | 72 |
| Shared Gate input quantization | 0.562 | 72 |
| Gate projection | 1.887 | 36 |
| Up input quantization | 0.000 | 0 |
| Up projection | 1.870 | 36 |
| SwiGLU | 0.258 | 36 |
| Down input quantization | 0.564 | 72 |
| Down projection | 3.206 | 36 |
| FFN residual | 0.257 | 36 |

Against the C9a P64 profile, deferred attribution moved from 27.991 ms to
27.828 ms and the fine-grained FFN stage total moved from 9.979 ms to about
9.519 ms. These are attribution-level comparisons rather than clean
throughput claims, but they are directionally consistent with removing one
Gate/Up quantization sequence. The shared path also removes 72 FFN-stage
dispatches and 72 total dispatches from the 1625-dispatch C9c control.

## 4. Interpretation

C9c's expected structural effect is visible: Gate/Up input quantization is
represented by one quantization sequence and the Up quantization stage has
zero dispatches. Total dispatches are 1553, down from the C9c control's 1625.
The shared reuse change did not alter attention scaling, allocation state, or
KV/cache behavior.

The current ranking is:

1. **FFN projection work — 6.963 ms.** This is the largest single family, but
   standalone Down geometry has already been rejected; future work should be
   pipeline-level or a measured common-kernel improvement.
2. **Attention — 4.935 ms at P64.** It remains context-dependent and will
   dominate at longer contexts, but its cooperative implementation is no
   longer pathological at this workload.
3. **Normalization — 2.978 ms / 289 dispatches.** This is the strongest
   structural candidate: many small operations remain, although their device
   time must be isolated before fusion.
4. **Conversion — 2.278 ms / 324 dispatches.** It is a substantial materialize-
   and-convert family, but precision boundaries must remain unchanged.
5. **LM head — 2.876 ms.** It is large but already one dispatch; target only
   after checking whether its kernel is the measured limiter.

## 5. Decision

```text
CLOSED — measurement-only; no production behavior changed
```

The next experiment should be selected from the normalization/conversion
small-kernel structure versus FFN pipeline cost, with one hypothesis and a
fresh A/B baseline. No C10b implementation is selected by this profile alone.

## 6. Follow-up

Preserve the shared Gate/Up Q8 policy and use this profile as the new baseline.
Do not reuse the stale C9a ranking for the next optimization.
