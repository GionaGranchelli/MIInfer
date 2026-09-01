# EXP-0033 — M5-C11a production and llama.cpp differential baseline

**Status:** CLOSED — measurement-only baseline
**Milestone:** M5
**Date:** 2026-09-01
**MIInfer policy:** shared Gate/Up Q8 reuse, separate C10c fusion
**Model:** Qwen3-8B Q4_0 (`Qwen3-8B-q4_0-b968826d.gguf`)

## 1. Hypothesis

After the C9c production KEEP and C10c rejection, the remaining MIInfer
performance gap should be explained by the current production profile and a
fresh same-machine llama.cpp control, rather than by another assumed
normalization or fusion opportunity.

## 2. Scope

This experiment changes no MIInfer production code. It refreshes the accepted
trace-free path and runs the pinned gfx906 llama.cpp reference with the same
model on the same MI50. The standard llama.cpp benchmark and the exact raw
token continuation are reported separately because their workload and timing
contracts differ.

## 3. Environment

```text
GPU: AMD Instinct MI50 / gfx906, one visible device
Model SHA256: 458634762bea7dbe19f3ce0614465bafd15ee90e815229c427043afcf195d628
MIInfer commit: af6b4337ce9e
Reference repository: milpster/gfx906-llama-cpp
Reference commit: 6e4ef6c1a553b8f61ad77bba18e9ca05aa677295
ROCm: 7.1.52802-9999
HIP compiler: clang 20.0.0.rocm
CPU: Intel(R) Xeon(R) CPU E5-2680 v3 @ 2.50GHz
```

MIInfer measurement artifacts captured approximately 925–930 MHz SCLK and
350 MHz MCLK during the measured decode phase. The fresh llama.cpp benchmark
telemetry reached approximately 1725 MHz SCLK and 1000 MHz MCLK during its
measured cases. This makes the headline rates a qualified differential, not
a clock-matched performance claim.

Raw MIInfer artifacts:

```text
bench/results/20260901T-c11a-minfer/
bench/results/20260901T-c11a-position-audit/20260901T193605Z-415258/
```

Raw external artifacts:

```text
/home/fedora-workstation/Development/mi50-artifacts/qwen3-8b-llama-c11a-20260901/
```

## 4. MIInfer production baseline

The benchmark used prompt ID `14990`, eight warmup tokens, 64 measured
decode forwards, five iterations per process, and three strictly serial runs.
Rejected C10c fusion was disabled and C9c shared reuse was enabled.

| Run | Decode ms | Decode tok/s |
|---:|---:|---:|
| 1 | 1164.044499 | 54.980716 |
| 2 | 1159.861791 | 55.178988 |
| 3 | 1162.080537 | 55.073636 |
| **mean** | **1161.995609** | **55.077780** |

All three runs produced the same 64-token sequence beginning
`8,341,286,470,330,9707,11,330`. Outputs were finite and deterministic.
Release CTest passed **19/19**.

## 5. MIInfer position audit

The accepted production path was audited at positions 1 and 64:

| Position | Wall ms | Whole-token GPU ms | Deferred GPU ms | Dispatches |
|---:|---:|---:|---:|---:|
| 1 | 15.265 | 15.461 | 23.553 | 1553 |
| 64 | 19.797 | 19.924 | 27.778 | 1553 |

At P64 the deferred category ranking is:

| Category | GPU ms | Dispatches |
|---|---:|---:|
| FFN projection | 6.956 | 108 |
| Attention | 4.937 | 36 |
| Quantization | 3.340 | 433 |
| Normalization | 2.947 | 289 |
| LM head | 2.876 | 1 |
| Conversion | 2.270 | 324 |
| Q/K/V projection | 1.696 | 108 |
| Residual | 0.517 | 72 |
| RoPE | 0.605 | 72 |
| Activation | 0.257 | 36 |
| KV store | 0.269 | 36 |
| GPU argmax | 0.277 | 1 |

The deferred categories are attribution data and sum above the whole-token
event because detailed scopes overlap and perturb execution. Clean wall and
whole-token GPU events remain the end-to-end timing truth. The path has zero
temporary allocations, 589,828 bytes of residual copy accounting, 38
synchronization sites, and 1,553 dispatches per token.

## 6. Fresh llama.cpp control

The pinned reference was run with full GPU offload and the same model:

```bash
llama-bench \
  -m /path/to/Qwen3-8B-q4_0-b968826d.gguf \
  -p 512,0 -n 0,128,256 -r 5 -b 2048 -ub 512 -ngl 99 -t 24 -o jsonl
```

| Workload | Mean tok/s | Stddev | Raw tok/s |
|---|---:|---:|---|
| PP512 | 984.552 | 13.727 | 960.037, 989.343, 991.412, 990.912, 991.057 |
| TG128 | 91.875 | 0.194 | 91.548, 92.031, 92.000, 91.931, 91.865 |
| TG256 | 91.692 | 0.097 | 91.713, 91.779, 91.758, 91.535, 91.676 |

The standard TG rates are approximately 1.67x the refreshed MIInfer rate, but
the clock states differ as recorded above. The closest exact raw-token control
retained from EXP-0012 reported 81.43 tok/s for 64 incremental `hello` tokens
at the low-clock regime, versus the earlier MIInfer control at 54.98 tok/s.
That 1.48x comparison is more workload-comparable but is not a fresh C11a
pair.

## 7. Interpretation and decision

The accepted path has eliminated the former attention-scaling, allocation,
static-upload, and tiny-KV-copy pathologies. FFN projection is the largest
current measured category at P64, followed by attention. Normalization,
quantization, and conversion remain substantial, but C10c demonstrated that
blindly combining those stages can reduce gfx906 device efficiency.

The external control demonstrates a substantial optimization opportunity and
near-flat standard TG128/TG256 behavior. It does not, by itself, identify one
MIInfer family because the external run reached a higher clock state and uses
its own graph-backed execution path.

**CLOSED — measurement-only.** No MIInfer production behavior changed.

The next bounded experiment is **M5-C11b — exact-shape FFN GEMV differential**:
compare MIInfer Gate, Up, and Down Q4_0 × Q8_1 primitives against the available
gfx906 MMVQ implementation at the exact Qwen3 shapes, then test at most one
mechanism-specific candidate if the differential identifies one. FFN is chosen
because it is the largest measured family; this is not a claim that the
end-to-end gap is exclusively an FFN issue.

C11b must preserve the current Q8 representation, FP32 output contract, and
64-token trajectory gate. It must separate kernel-level improvement from
whole-token impact and must not begin with another blind geometry sweep, fusion,
or graph capture.

## 8. Artifacts and follow-up

* MIInfer runs: `bench/results/20260901T-c11a-minfer/`
* MIInfer P1/P64 audit:
  `bench/results/20260901T-c11a-position-audit/20260901T193605Z-415258/`
* External benchmark:
  `/home/fedora-workstation/Development/mi50-artifacts/qwen3-8b-llama-c11a-20260901/`
* Historical exact raw-token comparison: `experiments/EXP-0012-llama-qwen3-comparison.md`

Before publishing a direct speed ratio, repeat the external and MIInfer
workloads with a consistent clock state. Until then, use the standard llama.cpp
figures as directional context and the exact raw-token result as historical
workload context.
