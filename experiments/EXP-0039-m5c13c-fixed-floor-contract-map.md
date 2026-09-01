# EXP-0039 — M5-C13c fixed-floor contract map

**Status:** CLOSED — measurement-only; no implementation selected  
**Milestone:** M5  
**Date:** 2026-09-02  
**Model:** Qwen3-8B Q4_0 (`Qwen3-8B-q4_0-b968826d.gguf`)

## 1. Question

Which fixed-floor operation families have an exact, evidence-backed
MIInfer/llama.cpp comparison contract, and which remain incomparable because
their representation or execution contract differs?

## 2. Method

This is a contract-map audit, not a new implementation or benchmark. It
cross-references the accepted MIInfer P1 profile from
[`EXP-0037`](EXP-0037-m5c13a-fixed-floor-profile.md) with the direct primitive
and external-reference evidence in [`EXP-0008`](EXP-0008-direct-mmvq.md),
[`EXP-0009`](EXP-0009-kv-geometry.md), [`EXP-0034`](EXP-0034-m5c11b-exact-shape-ffn-differential.md),
and [`EXP-0038`](EXP-0038-m5c13b-lm-head-contract-audit.md).

An operation is marked **Yes** only when the available evidence covers the
same semantics, shape, and representation at the compared boundary. A whole
family is not considered comparable merely because both runtimes contain an
operation with the same name.

## 3. Fixed-floor contract map

MIInfer costs are P1 deferred category/stage timings from C13a. They overlap;
they are not additive wall-clock components. The P1 clean wall was 15.032 ms,
the whole-token GPU event was 15.071 ms, and the fixed wall-minus-attention
floor was approximately 14.7 ms/token.

| Family | MIInfer production contract | llama.cpp evidence | Comparable? | MIInfer P1 cost | Disposition |
|---|---|---|:---:|---:|---|
| FFN Gate/Up/Down | Q4_0 × Q8_1, FP32 accumulation, production output boundary | Q4_0 × Q8_1 MMVQ at exact Qwen3 shapes | Yes* | 6.969 ms combined | Cleared by C11b; no new FFN target |
| Q/K/V projections | Q4_0 × Q8_1, exact Qwen3 shapes | Matching Q4_0 × Q8_1 MMVQ primitive; K/V follow-up in EXP-0009 | Yes* | 1.648 ms | Cleared; no new geometry target |
| O projection | Q4_0 × Q8_1, exact Qwen3 shape | Matching Q4_0 × Q8_1 MMVQ primitive | Yes* | 0.819 ms | Cleared by the direct projection comparison |
| LM head | Q6_K × Q8_K → FP32 logits | Available pinned path is Q6_K × Q8_1 → FP32 | No | 2.943 ms | C13b negative result; do not force a compatibility path |
| RMSNorm | FP32 reduction/scale with explicit consumer-specific materialization | No retained exact primitive/representation comparison | No | 2.856 ms | Needs a valid external contract before differential work |
| Quantization | Q8_1 for projections; Q8_K for LM head, with established metadata/rounding | Q8_1 is exposed for MMVQ; no matching whole-family timing contract | Partial | 3.179 ms | Gate/Up Q8_1 reuse kept; broad family remains unranked |
| Conversion/materialization | Explicit F32↔F16 boundaries where required by production semantics | Backend-specific materialization; exact boundary not established | No | 2.168 ms | C10c fusion rejected; no generic fusion |
| RoPE / KV operations | Qwen3 RoPE, GQA, persistent cache layout, validated position semantics | No retained exact-shape/representation differential | No | 0.534 / 0.256 ms | Semantics validated; not a differential target |
| Cached attention | Cooperative cached attention with its validated reduction/materialization contract | llama.cpp uses a different attention execution contract | No | 0.450 ms at P1 | Production path kept; scaling separately characterized |
| GPU greedy argmax | GPU argmax with first-index tie behavior, 4-byte token D2H | External sampling path is not the same exposed contract | No | 0.279 ms | Structural cleanup kept; not a remaining major target |

\* The direct low-level comparisons record a historical MIInfer FP16 versus
MMVQ FP32 output-boundary difference. The input/weight quantization contract,
shape, and measured primitive path are nevertheless the accepted C11b
comparison; the result cleared these projection families rather than selecting
a port.

## 4. Eligible rows and ranking

The only rows with an established direct same-contract differential are the
Q4_0 × Q8_1 projection families. They are already cleared: MIInfer was
approximately tied or faster for the relevant Gate, Up, Down, Q, and O paths,
and EXP-0009 addressed the historical K/V geometry gap. No uncleared row has
both a valid external contract and a demonstrated MIInfer differential in this
audit.

The remaining fixed-floor rows therefore cannot be ranked against llama.cpp by
kernel latency without first obtaining a matching contract. Their MIInfer-only
costs are useful for future measurement, but are not evidence that llama.cpp
is faster for that family.

## 5. Decision

```text
C13c fixed-floor contract map: CLOSED
Production code: unchanged
LM-head Q8_1 compatibility path: not added
Next implementation experiment: not selected
```

The next optimization must follow a new same-contract measurement or a
separately justified MIInfer-owned experiment with a bounded end-to-end
hypothesis. Do not replace CPU-reference overfitting with llama.cpp-reference
overfitting.

## 6. Follow-up

Maintain the accepted production baseline and use this map to reject invalid
cross-contract differentials. Reopen a family only when its semantics, shape,
and representation can be compared directly and its whole-token leverage is
large enough to justify the work.
