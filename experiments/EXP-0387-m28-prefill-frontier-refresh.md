# EXP-0387 — M28 full-model prefill frontier refresh

## Question

Refresh the current full-model prefill frontier after EXP-0376 through
EXP-0386, close the rejected B64 direct-wide family, and select exactly one
measured next PRIMARY. This is measurement/decision only; no optimization
candidate was implemented.

## Basis and environment

EXP-0378 through EXP-0386 show that direct whole-model B64 recurrent-wide
execution runs, but produces distributed semantic drift. L0/L1/L2 scalar
hybridization improves similarity without restoring the token. B4 remains
blocked.

| Item | Value |
|---|---|
| Model | Qwen3.8-27B-Q4_K_M.gguf; SHA `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169` |
| GPU | MI50-class gfx906, 32 GB HBM2 |
| Clocks/policy | SCLK 1606 MHz, MCLK 1000 MHz, 225 W; external fan full speed |
| HIP/compiler | HIP 7.1.52802-9999; clang 20.0.0.rocm |
| MIInfer | `3da990036c6705f7fc67a16840b1f2a28062b7b8`, `build/mi50-release` |
| mx reference | `2e9d29fe736969160f17476ec6f0a6298cee6966` |

Clean runs used only the qualified `m25_hi_qualified` route and repeated-B128
scheduler; all EXP-0380–0386 selectors and profiler were disabled. The mx
benchmark uses its normal synthetic input, so prompt text differs although
model, quantization, GPU, and token counts are matched.

## Clean timing matrix

| Prompt | MIInfer decomposition | MIInfer ms | MIInfer tok/s | mx ms | mx tok/s |
|---:|---|---:|---:|---:|---:|
| 512 | 1×B512 | 3450.08 | 148.40 | 2298.61 | 222.75 |
| 1024 | 2×B512 | 5299.90 | 193.21 | 4329.02 | 236.54 |
| 2048 | 4×B512 | 12024.34 | 170.32 | 8665.67 | 236.34 |
| 4096 | 8×B512 | 24441.10 | 167.59 | 17494.25 | 234.13 |
| 8192 | 16×B512 | 51840.02 | 158.02 | 35730.29 | 229.27 |
| 640 | 1×B512 + 1×B128 | 16031.40 | 39.92 | 2772.13 | 230.87 |
| 896 | 1×B512 + 3×B128 | 42464.99 | 21.10 | 3833.80 | 233.71 |
| 1022 | 1×B512 + 3×B128 + 126 scalar | 106772.40 | 9.57 | 4380.45 | 233.31 |
| 2174 | 4×B512 + 3×B128 + 126 scalar | 76254.75 | 28.51 | not rerun | — |
| 4222 | 8×B512 + 3×B128 + 126 scalar | 87694.19 | 48.14 | not rerun | — |

P640/P896 have zero scalar residual. P1022/P2174/P4222 report 126 residual
tokens and 64 scalar layer-run calls / 8064 scalar layer-run tokens.

## Attribution

The required EXP-0376 event timing on a clean P1022 diagnostic run was:

| Chunk | Base | Count | GPU ms | Host wall ms | Host/GPU gap | Increment vs first B128 |
|---|---:|---:|---:|---:|---:|---:|
| B512 | 0 | 512 | 3105.16 | 676.71 | -2428.46 | — |
| B128 #1 | 512 | 128 | 14725.80 | 15834.40 | 1108.57 | baseline |
| B128 #2 | 640 | 128 | 13850.70 | 14087.50 | 236.79 | GPU -875.10; host -1746.90 |
| B128 #3 | 768 | 128 | 12851.50 | 12755.50 | -95.96 | GPU -1874.30; host -3078.90 |

The event timing confirms that later B128 invocations do not jump in cost by
invocation number; all three are already very expensive, and their GPU time
does not grow with base position in this run. The negative B512 gap reflects
the event/call-boundary relationship and is not treated as a host-speed claim.
The event run recorded `alloc_delta=0`, `total_bytes_delta=0`, and
`live_bytes_delta=0` for every timed chunk.

P896 is already `42.465 s` with zero scalar residual. P1022 has the same
B512 + 3×B128 prefix and is `106.772 s`; the matched delta is approximately
`64.307 s`, or `60.2%` of P1022 wall time. P896 minus P512 is approximately
`39.015 s` for the three repeated B128 chunks, or `91.9%` of P896 wall time.
These are production-contract deltas, not kernel-only claims.

The clean profile at P512, P1024, and P1022 showed the same recurrent and
attention invocation families for aligned work and the expected three B128
wide-prefill invocations plus scalar tail at P1022. Its scopes overlap and
were not summed. No B64 route, capture, state export, token dump, repeated
model-sized allocation, or weight-repack/setup operation appeared. The clean
wall time remained close to profiled wall time; no evidence makes host/runtime
overhead the primary cause, but scalar-layer host/GPU time is not separately
measured.

For causal attention, B128 prefixes grow from roughly 513–640 at base 512 to
641–768 at base 640 and 769–896 at base 768. That predicts moderate scaling,
not the observed 5–10× request collapse. The P896 result falsifies
context-length-only attribution. There is no internal route difference in the
clean counters: `partial_b64=0`, with scalar work only for the 126-token tail.

| Frontier | Measured ceiling | Decision |
|---|---:|---|
| Repeated B128 tail | 39.015 s; 91.9% of P896 | production-tail problem |
| 126-token scalar residual | 64.307 s; 60.2% of P1022 | production-tail problem |
| Aligned MIInfer-vs-mx gap | 0.971 s at P1024; 18.3% of MIInfer P1024 | not enough for kernel selection |
| Attention / FFN | no non-overlapping end-to-end ceiling | not authorized |
| Host/runtime setup | no repeated model-sized setup observed | not primary |

## B64 direct-wide disposition

**REJECTED — CURRENT M28 FRONTIER.** The premise was to compose existing
count-64 recurrent-wide primitives to remove 64 tokens from the sub-B128
scalar tail. EXP-0378 through EXP-0386 prove successful execution and healthy
local numerical contracts but distributed whole-model semantic divergence; the
L0–L2 hybrid does not restore semantics. Reopen only for a materially new
semantic strategy with a credible measured end-to-end ceiling. Another group,
layer, tolerance, scalar map, or B64 parameter sweep is not new evidence.
B4 remains blocked.

## Decision

**QUALIFY.** The aligned frontier has an ordinary aggregate MIInfer-vs-mx gap;
arbitrary-tail production behavior is dominated by repeated B128 plus the
residual contract, not context length alone.

The one next PRIMARY is:

> Measure and qualify a materially different, semantically safe arbitrary-
> length residual architecture, starting with an exact P1022 contract and an
> end-to-end ceiling. Do not reopen direct B64-wide or begin B4 work.

No new GPU math kernel or residual scheduler optimization was implemented.

Experiment commit: `76792de` (`EXP-0387 refresh M28 prefill frontier`).
Graph SHA: `cff7ac372a0aa16728774640c8a6c8dabc364e5243ad4c4b22fcd9c74f94090d`.
Final provenance commit is the follow-up documentation commit.
