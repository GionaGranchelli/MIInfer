# EXP-0352 — M26-B historical decode-floor differential

**Status:** ACTIVE — diagnostic entry gate; no optimization authorized

## Question

What accounts for the approximately `24.5 ms/token` difference between the
historical M8/M9 result (`~33 ms/token`) and the current M25/M26 P512 result
(`~59 ms/token`) under a controlled, apples-to-apples contract?

M26-B closes only when at least 90% of the wall-time delta is attributed, or
the historical result is conclusively shown to be non-comparable.

## Known measurements

| Path | Model | Context | Generation | Result |
| --- | --- | ---: | ---: | ---: |
| M8 qualified | Qwen3.8-27B-Q4_K_M | TG64 | 64 | `33.01 ms/token`, `30.29 tok/s` |
| M9 qualified | Qwen3.8-27B-Q4_K_M | TG64 | 64 | `33.01 ms/token` baseline |
| M26 current | Qwen3.8-27B-Q4_K_M | P512 | 128 | `57.50 ms/token`, `17.39 tok/s` |
| M26 current | Qwen3.8-27B-Q4_K_M | 12K | 128 | `63.33 ms/token`, `15.79 tok/s` |

These values are not yet a regression claim. M8/M9 and M26 may differ in
execution contract, graph coverage, state preparation, and timing boundary.

## B1 — historical contract reconstruction

The source records establish the following historical facts:

| Field | Historical M8/M9 | Current M26 | Status |
| --- | --- | --- | --- |
| Model/hash | Qwen3.8-27B-Q4_K_M; hash recorded in M8 | same hash | SAME |
| GPU | MI50/gfx906, 60 CU | MI50/gfx906, 60 CU | SAME |
| Clocks | 1606/1000 MHz, 225 W | must record per run | VERIFY |
| Build | Release | Release | VERIFY compiler/toolchain |
| Decode length | TG64/TG128 | 128 tokens | DIFFERENT naming; normalize |
| Context | short fixed state | 512 existing tokens | VERIFY exact state |
| Graph | qualified full decode graph | interactive graph limit 4096; 12K graph disabled | DIFFERENT |
| State/KV | persistent decode state | current persistent state | VERIFY layout and preparation |
| Timing | decode benchmark boundary | `stats.decode_ms` plus wall latency | VERIFY inclusion |
| LM head/argmax | documented as part of decode | current path includes both | VERIFY |
| Sampling | historical benchmark behavior | current greedy/sampling behavior | VERIFY |
| Allocations | zero during decode | must record | VERIFY |

Unknown or merely inferred fields remain open until recovered from the
historical benchmark source, commit, or raw result.

## B2 — current contract

The current reproducible command is:

```text
MIINFER_PRESET=m25_interactive \
build/mi50-release/miinfer run MODEL --context 16384 \
  --decode-curve --curve-context 512 --curve-iterations 5 --no-stream
```

The current five-run curve reports `57.501 ms/token` at 512 and
`63.333 ms/token` at 12K. The validated diagnostic profile at the first
post-prompt token reports `58.789 ms` accounted versus `59.124 ms` wall at
512, and `64.179 ms` accounted versus `63.669 ms` wall at 12K. This establishes
the current profiler boundary, not equivalence with M8/M9.

## B3 — required comparison matrix

Run and retain raw output for:

1. historical code + historical workload;
2. current code + historical workload;
3. historical code + current workload where the old interface permits it.

For each variant record model hash, clocks, ROCm/compiler, flags, context,
warmup, generated tokens, graph state, state preparation, timing boundary,
LM-head/sampling inclusion, allocations, and VRAM.

If an old commit cannot build or run on the current machine, record the exact
failure and reproduce its operations with the closest equivalent current
harness. Do not silently substitute a different workload.

## B4 — differential attribution

The matched report must contain historical ms/token, current ms/token, delta,
and regression share for:

```text
recurrent QKV; GDN/state update; recurrent Gate/Up/SwiGLU; recurrent Down;
attention Q/K/V; Q/K norm and RoPE; KV append; QK/softmax/V accumulation;
attention O; RMSNorm; quantization/conversion; residuals; LM head; sampling;
graph/runtime; host synchronization; other/unattributed
```

Also report dispatches/token, graph launches/token, synchronizations/token,
and repeated per-layer costs. Normalize material differences, for example:
`12 ms / 48 recurrent layers = 250 us/layer`.

## B5 — fast-path audit

Audit every M8/M9 optimization as `ACTIVE`, `REPLACED`, `DISABLED`,
`UNREACHABLE`, `REMOVED`, or `UNKNOWN`: Wave64 Q8 quantization, fused Q8
epilogues, SwiGLU shuffle reduction, SIMD Q6_K unpack, native Q4/Q5/Q6 GEMV,
resident weights, persistent buffers, and full decode graph coverage.

No `UNKNOWN` item may remain at gate close.

## Decision gates

- **B1:** reproduce the historical result within ±5%, or document why it is
  non-comparable.
- **B2:** classify every material contract difference.
- **B3:** explain at least 90% of the historical-to-current wall delta.
- **B4:** divide the delta into contract difference, intentional work,
  recoverable regression, unavoidable cost, and remainder <=10%.

No M26-C kernel optimization begins before B1–B4. The full-attention output/KV
track remains capped at its expected `4–6 ms/token`; fixed-floor work is chosen
only from this differential.

## Exit table

| Family | Historical | Current | Delta | Share | Classification |
| --- | ---: | ---: | ---: | ---: | --- |
| recurrent QKV | pending | pending | pending | pending | pending |
| GDN/state | pending | pending | pending | pending | pending |
| FFN | pending | pending | pending | pending | pending |
| attention/KV | pending | pending | pending | pending | pending |
| quantization/conversion | pending | pending | pending | pending | pending |
| graph/runtime/sync | pending | pending | pending | pending | pending |
| LM head/sampling | pending | pending | pending | pending | pending |
| contract/other | pending | pending | pending | pending | pending |

**Decision:** RETEST — diagnostic gate open; optimization prohibited.

## Audit execution — 2026-09-15

| Item | Evidence | Result |
| --- | --- | --- |
| Current HEAD | `dd7b469647693cb46caacf8aa3e7bf31333805a4` | recorded |
| Worktree | clean before this diagnostic update | recorded |
| Model SHA-256 | `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169` | PASS |
| Required fixture | `/tmp/m6a273-reference-p12` absent; `/tmp/m6a273-reference` also absent | BLOCKED |
| Historical worktree | `/tmp/miinfer-m26b-m8` at `ff7aefffbe33ea386aeef0e12d1a8623fc6ff58d` | PASS |
| Historical legacy Release build | `miinfer-m6a21-qwen35-gpu-hybrid-block` | PASS |
| HIP toolchain | HIP `7.1.52802-9999`, clang `20.0.0.rocm`; GCC `16.2.1` | recorded |
| GPU | MI50 / gfx906 / 60 CU | recorded |
| Audit clocks | SCLK `930 MHz`, MCLK `350 MHz`; power cap `225 W` | INVALID for qualification |

The required historical/current `--bench64` A/B was not run because the
fixture is absent and the audited clock state is not comparable to the
qualified 1606/1000 MHz state. No historical/current differential, regression
classification, or M26-C target is claimed from this audit.

**Audit decision:** M26-B RETEST — differential not sufficiently explained.

## Reconstructed-fixture probe — 2026-09-16

The original `/tmp/m6a273-reference-p12` bundle could not be recovered. Since
the absence was confirmed, a separately named replacement was generated from
llama.cpp commit `c0bc8591e8815c63cb01dd3f051a8b0df02501c` using the tracked
fixture exporter. Validation passed; this is not claimed to be byte-identical
to the historical `-p12` bundle.

| Variant | Result | Correctness/runtime evidence |
| --- | --- | --- |
| `HISTORICAL_LEGACY` — `ff7aeff` + reconstructed fixture | `33.227 ms/token`, `30.096 tok/s` | replay PASS; allocations 0 |
| `CURRENT_LEGACY` — current HEAD + same reconstructed fixture | no timing result | abort: HIP invalid argument at `tools/qwen35_gpu_pipeline.hpp:626` during device-to-host fingerprint copy |

The historical harness reproduces the expected 30.29 tok/s result within 5%.
The current legacy harness is not comparable yet because it aborts before
reporting a measurement against this replacement fixture. This failure is
recorded as an execution/fixture compatibility issue, not classified as a
performance regression.

The required original `-p12` fixture remains the next prerequisite for closing
the historical/current differential. No kernel or quantization changes were
made.

**Probe decision:** M26-B RETEST — differential not sufficiently explained.

## Current legacy rerun — 2026-09-16

The comparison harness fingerprint was corrected to use the actual current KV
element size. This changes diagnostic copying only; it does not alter model
execution or timing semantics.

| Variant | Median | Throughput | Replay | Decode allocations |
| --- | ---: | ---: | --- | ---: |
| `HISTORICAL_LEGACY` — `ff7aeff` + reconstructed fixture | `33.227 ms/token` | `30.096 tok/s` | PASS | 0 |
| `CURRENT_LEGACY` — current HEAD + reconstructed fixture | `31.385 ms/token` | `31.863 tok/s` | PASS | 0 |

Both runs used the M8 flag vector, `MIINFER_DEVICE_TOKEN_CHAIN=0`, one warmup,
five measured samples, the same model, reconstructed fixture, and manual
1606/1000 MHz clocks. The current legacy path is `1.842 ms/token` faster than
the historical path. Therefore the shared legacy GPU pipeline does not contain
the approximately 24.5 ms/token loss seen in the current M25 runtime result.

The remaining differential is in the current runtime contract and/or route
selection. The current M25 P512 result is `57.50 ms/token`; compared with the
current legacy result, the observed gap is approximately `26.12 ms/token`.
This is not yet subdivided into route, graph, timing, and runtime components.

**Updated decision:** M26-B RETEST — shared-pipeline regression ruled out;
runtime/route differential still requires attribution.

## Runtime route isolation — 2026-09-16

With manual 1606/1000 MHz clocks and the same 512-context, 128-token curve
semantics:

| Current runtime route | Median ms/token | Throughput |
| --- | ---: | ---: |
| `m25_hi_qualified` | `532.856` | `1.877 tok/s` |
| `m25_interactive` | `83.759` | `11.939 tok/s` |
| current legacy harness | `31.385` | `31.863 tok/s` |

The interactive Mx route saves `449.097 ms/token` versus the non-Mx serving
route, confirming that route selection is a material performance factor. It
still trails the current legacy harness by `52.374 ms/token`. The curve's
`decode_ms` excludes graph-capture bookkeeping but includes the first token;
no conclusion is made yet about the remaining graph, runtime, or timing-boundary
components.

**Route-isolation decision:** M26-B RETEST — Mx route is necessary but does
not explain the complete legacy/runtime differential.
