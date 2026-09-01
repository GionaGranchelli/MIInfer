# EXP-0037 — M5-C13a fixed-cost floor profile

**Status:** CLOSED — measurement-only; C13b selected  
**Milestone:** M5  
**Date:** 2026-09-01  
**Model:** Qwen3-8B Q4_0 (`Qwen3-8B-q4_0-b968826d.gguf`)

## 1. Question

After C12 established predictable linear cached-attention scaling, where does
the approximately 14.7 ms/token context-independent floor live at short
context, and which one remaining family should be investigated next?

## 2. Method and environment

The accepted production path was measured without changing kernels or runtime
behavior. It used shared Gate/Up Q8 reuse, the accepted cooperative attention
kernel, GPU greedy argmax, and `stable_peak` on the MI50:

```text
SCLK 1725 MHz
MCLK 1000 MHz
positions 1, 2, 4, 8
attention selection: parallel (production default)
FFN Q8 reuse: shared (production default)
```

The raw audit is retained at:

```text
bench/results/20260901T-c13a-floor/position-audit.json
```

The current clean production tree was `1f1d6e28fdc0` with no uncommitted
changes. The audit is measurement-only; no production code or selection
changed.

## 3. Position results

| Position | Wall ms | Whole-token GPU ms | Deferred ms | Attention ms | Fixed wall-minus-attention ms |
|---:|---:|---:|---:|---:|---:|
| 1 | 15.032 | 15.071 | 22.832 | 0.450 | 14.582 |
| 2 | 15.342 | 15.186 | 22.700 | 0.522 | 14.820 |
| 4 | 15.431 | 15.425 | 22.991 | 0.663 | 14.768 |
| 8 | 15.712 | 15.656 | 23.107 | 0.949 | 14.763 |

The fixed wall-minus-attention component is stable at approximately 14.7
ms/token. This agrees with the independent C12b endpoints:

```text
P64:   19.665 - 4.932 = 14.733 ms
P1024: 88.678 - 74.071 = 14.606 ms
```

The whole-token GPU event tracks wall time closely, so this profile does not
show a large unaccounted CPU submission or GPU-idle gap.

## 4. P1 production attribution

Deferred category timings overlap and are not additive wall-clock components;
the clean wall and whole-token event remain the authoritative totals.

| Category | GPU ms | Dispatches |
|---|---:|---:|
| FFN projection | 6.969434 | 108 |
| Quantization | 3.179193 | 433 |
| LM head | 2.942877 | 1 |
| Normalization | 2.855519 | 289 |
| Conversion | 2.167838 | 324 |
| Q/K/V projection | 1.647519 | 108 |
| Residual | 0.479039 | 72 |
| RoPE | 0.534399 | 72 |
| Attention | 0.450400 | 36 |
| Activation | 0.244799 | 36 |
| KV cache | 0.255520 | 36 |
| Argmax | 0.278560 | 1 |
| Embedding | 0.008000 | 1 |

Structural counters were unchanged at every audited position:

```text
1553 dispatches/token
38 synchronization sites/token
0 allocations/token
589828 residual copy bytes/token
```

## 5. Interpretation

The short-context floor is not explained by attention growth, allocations,
copy expansion, dispatch growth, or an observable wall-versus-GPU idle gap.
FFN projection remains the largest named family, but C11b already found
MIInfer approximately tied or faster than the pinned gfx906 MMVQ path for the
Gate, Up, and Down shapes. Reopening FFN GEMV geometry is therefore not
justified by this profile.

Among the remaining uncleared fixed-cost families, the LM head is a large,
isolated 2.943 ms P1 stage with one dispatch and an exact Q6_K×Q8_K GEMV
contract. It is a cleaner differential target than the broad quantization,
normalization, or conversion categories, especially after C10c showed that
blind boundary fusion can regress the production path despite reducing
dispatches.

The pinned llama.cpp shape control measured approximately 90.817 tok/s for
PP1/TG64 at the same stable-peak clocks, or about 11.01 ms per token. This is
directional because the control is not token-identical, but it confirms a
material fixed-cost differential remains to be explained.

## 6. Decision

```text
C13a fixed-floor attribution: KEEP
Production behavior: unchanged
Next target: M5-C13b exact-shape LM-head differential
```

## 7. Follow-up — M5-C13b

Run a measurement/forensics experiment before implementing a replacement.
Compare the production MIInfer LM-head Q6_K×Q8_K GEMV with the strongest
reproducible gfx906 reference at the exact Qwen3 vocabulary and hidden-size
shape, stable-peak clocks, and identical input representation where possible.
Record latency, effective bandwidth, launch geometry, resource usage, and
whether the result is limited by the LM-head kernel or its input preparation.

Do not change production selection, quantization, accumulation semantics, or
the final GPU argmax in C13b. Only a clear exact-shape differential should
justify a later implementation experiment.
