# EXP-0040 — M5-C14a fixed-floor execution map

**Status:** CLOSED — measurement-only; no implementation selected  
**Milestone:** M5  
**Date:** 2026-09-02  
**Model:** Qwen3-8B Q4_0 (`Qwen3-8B-q4_0-b968826d.gguf`)

## 1. Question

Which operations execute for one MIInfer Qwen3 decode token, what
representations cross each boundary, and which differences from the retained
llama.cpp evidence are implementation, representation, work-elimination, or
scheduling/fusion differences?

## 2. Method and limits

This is a measurement/map slice. It uses the current production layer path,
the P1 fixed-floor profile in [`EXP-0037`](EXP-0037-m5c13a-fixed-floor-profile.md),
the boundary attribution in `EXP-0031`, and the valid direct projection
comparisons in `EXP-0008`, `EXP-0009`, and `EXP-0034`.

The MIInfer costs below are deferred P1 category/stage timings unless marked
otherwise. They overlap and must not be added as wall-clock components. The
authoritative P1 totals are 15.032 ms clean wall, 15.071 ms whole-token GPU
event, and approximately 14.7 ms wall-minus-attention fixed floor.

`Not established` means the retained llama.cpp evidence does not expose the
same stage contract or enough data to claim fusion, reuse, or work elimination.
It is not evidence that llama.cpp does or does not perform that work.

## 3. Layer/token execution map

| Stage | MIInfer representation/path | llama.cpp retained evidence/path | Same work? | MIInfer cost | Quantizations / materializations / transitions | Dispatches | Difference class |
|---|---|---|:---:|---:|---|---:|---|
| RMSNorm | FP32 reduction and scale; consumer-specific output buffers | Same high-level operation not exposed as an exact primitive contract | Unproven | 2.856 ms category | FP32 norm output; later consumer conversions | included in 2.856 ms category | B/D — representation and boundary behavior unestablished |
| Q projection | Q4_0 × Q8_1 GEMV; FP32 accumulation, production output boundary | Q4_0 × Q8_1 gfx906 MMVQ at matching Qwen3 shape | Yes* | included in Q/K/V 1.648 ms | one projection-input Q8_1 path; output materialization | 1 projection dispatch | A — direct kernel comparison cleared |
| K projection | Q4_0 × Q8_1 GEMV; exact K/V geometry retained | Q4_0 × Q8_1 MMVQ; EXP-0009 addressed K/V geometry | Yes* | included in Q/K/V 1.648 ms | projection output then K head norm/RoPE | 1 projection dispatch | A — cleared under retained protocol |
| V projection | Q4_0 × Q8_1 GEMV; V enters cache unmodified after projection | Q4_0 × Q8_1 MMVQ | Yes* | included in Q/K/V 1.648 ms | projection output then one KV cache store | 1 projection dispatch | A/C — kernel cleared; cache materialization differential unestablished |
| RoPE | Qwen3 RoPE on Q and K after head norms; position-selected outputs | Exact external stage contract not retained | Unproven | 0.534 ms | Q and K RoPE outputs materialized before attention/cache | 72 boundary dispatches in P1 profile | B/D — representation/scheduling not comparable |
| KV append | Persistent FP32 K/V cache, GQA layout, one current-token store per head/value | Exact cache layout/store path not retained | Unproven | 0.256 ms | 36 layer cache operations; no per-head memcpy in production | 36 | C/D — work shape may differ, not measured externally |
| Cached attention | Cooperative cached attention; FP32-sensitive reduction and validated output boundary | Different backend attention execution contract | No | 0.450 ms at P1; 4.932 ms at P64 | reads persistent history; FP32 attention then F32→F16→F32 boundary | 36 | B/D — backend representation and reduction contract differ |
| O projection | Q4_0 × Q8_1 GEMV; exact Qwen3 shape | Q4_0 × Q8_1 MMVQ at matching shape | Yes* | 0.819 ms | attention FP32 round-trip precedes projection input preparation | 1 projection dispatch | A/B — primitive cleared; surrounding boundaries differ or are unestablished |
| FFN norm | FP32 RMSNorm and scale; shared Gate/Up Q8_1 input is production-selected | Exact external stage contract not retained | Unproven | included in normalization 2.856 ms | one shared Gate/Up Q8_1 quantization after required boundaries | 72 norm-related dispatches in P1 attribution | B/C/D — reuse known in MIInfer; external reuse unknown |
| Gate / Up | Q4_0 × Q8_1 GEMV; shared identical Q8_1 input | Matching Q4_0 × Q8_1 MMVQ shapes | Yes* | included in FFN projection 6.969 ms | one shared activation Q8_1 quantization for both consumers | 2 projection dispatches/layer | A/C — direct kernels cleared; MIInfer reuse is a KEEP |
| SwiGLU activation | `silu(gate) * up` into FP intermediate | Exact external stage contract not retained | Unproven | 0.245 ms | FP intermediate materialized for Down input | 36 | B/D — fusion candidate C9b was trajectory-unsafe |
| Down-input quantization | Existing Q8_1 contract after the established FP16 boundary | Q8_1 input for matching MMVQ primitive, stage timing not retained | Partial | included in quantization 3.179 ms | one Q8_1 quantization for Down input | 36 | B — representation boundary is comparable only at the primitive input |
| Down projection | Q4_0 × Q8_1 GEMV; long-K production geometry | Matching Q4_0 × Q8_1 MMVQ shape | Yes* | included in FFN projection 6.969 ms | consumes existing Down Q8_1 blocks; FP output materialized | 1 projection dispatch/layer | A — C8/C11b cleared standalone geometry |
| Layer residual/output | FP32 residual add into persistent ping-pong output | Exact external stage contract not retained | Unproven | included in residual work | direct layer-output handoff; no allocation/copy in production fast path | 72 P1 residual events | C/D — MIInfer materialization path is owned and specialized |
| Final norm | FP32 RMSNorm | Exact external stage contract not retained | Unproven | about 0.021 ms in boundary attribution | FP32 output then Q8_K final activation | 1 | B — no same-contract differential |
| LM head | Q6_K × Q8_K → FP32 logits | Pinned MMVQ exposes Q6_K × Q8_1 → FP32 | No | 2.943 ms | one Q8_K quantization; full logits remain on GPU for argmax | 1 | B — exact representation mismatch; C13b closed |
| Logits / argmax | GPU argmax with first-index tie semantics; 4-byte token result | External sampling/output path is not the same exposed contract | Unproven | 0.279 ms | no full-logit D2H in greedy production path | 1 argmax dispatch | C/D — MIInfer eliminated full-logit transfer |

\* The direct low-level projection comparisons retain a historical MIInfer
FP16 versus MMVQ FP32 output-boundary difference. They are accepted as
Q4_0×Q8_1 input/weight/shape comparisons for kernel differential purposes;
they do not prove identical whole-stage runtime graphs.

## 4. Difference inventory

### A — implementation difference, same contract

The Q4_0 × Q8_1 Q, K, V, O, Gate, Up, and Down primitive comparisons are the
only retained direct same-contract evidence. C11b and EXP-0009 cleared those
families for the current optimization path. No new FFN or projection port is
justified by C14a.

### B — representation difference

The clearest case is the LM head: MIInfer uses Q6_K × Q8_K while the pinned
llama.cpp path uses Q6_K × Q8_1. MIInfer also retains explicit FP32/FP16
materialization boundaries around attention and projection preparation. These
are not valid latency differentials until the external representation contract
matches.

### C — work-elimination difference

MIInfer has directly measured and retained several work eliminations: shared
Gate/Up Q8_1 quantization, persistent decode workspace, resident norm weights,
direct layer-output handoff, coarsened KV stores, and GPU-side greedy argmax.
The retained llama.cpp evidence does not expose a complete stage map, so no
additional llama.cpp-only work elimination is claimed here.

### D — scheduling/fusion difference

The runtimes clearly have different attention and materialization execution
contracts. C10c demonstrates that reducing dispatches without reducing useful
device work can regress MI50 performance: its fused path removed 108 dispatches
but was 5.217% slower and was rejected. No generic fusion conclusion follows
from the dispatch counts alone.

## 5. Fixed-floor budget

The fixed floor is approximately 14.7 ms/token, but deferred family timings
overlap. The defensible accounting is therefore:

| Bucket | Current evidence | Budget status |
|---|---|---|
| Necessary equivalent work | Q4_0 × Q8_1 projection families are directly cleared; P1 FFN projection is 6.969 ms and Q/K/V/O is 2.467 ms in deferred attribution | Named work, not additive wall cost |
| Representation-different work | LM head is 2.943 ms with an invalid Q8_K vs Q8_1 external differential; conversion is 2.168 ms with no exact external boundary | Candidate surface, not removable budget |
| MIInfer-only/redundant work | No new unambiguously redundant family remains after C9c and C6 cleanup | No defensible milliseconds claimed |
| Unclassified fixed work | Norm, activation, residual, RoPE/KV, and other backend-specific stages lack complete same-contract external evidence | Remaining parity budget is unknown |

C14a therefore does not claim that llama.cpp avoids any particular number of
milliseconds. It identifies which future measurements would be valid and
prevents treating incomparable representation or fused-boundary costs as a
proven software differential.

## 6. Decision

```text
C14a execution/contract map: CLOSED
Production code: unchanged
Valid uncleared same-contract differential: none identified
Next implementation experiment: not selected
```

The next experiment must either establish a matching external contract for a
remaining high-leverage family or present a narrowly bounded MIInfer-owned
work-elimination hypothesis. Do not add a compatibility path solely to create
a benchmark comparison.
