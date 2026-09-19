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

## Route-contract recovery probe — 2026-09-16

`generate_layer_major()` now uses the existing queued no-stream HIP graph chain
after layer-major prefill, matching the decode contract already used by
`generate_fresh()`. Streaming and graph-disabled generation retain the
per-token `step()` path. The MIInfer target builds successfully.

No runtime result is claimed: no Qwen3.8-27B GGUF is present in the workspace
for a MI50 run. The next measurement is the same reconstructed-fixture curve
with `MIINFER_PRESET=m25_interactive`, followed by correctness comparison.

**Probe decision:** M26-B RETEST — implementation landed; performance and
execution-contract differential remain unverified.

**Route-isolation decision:** M26-B RETEST — Mx route is necessary but does
not explain the complete legacy/runtime differential.

## Hardware readiness check — 2026-09-16

`miinfer doctor` passes on the available accelerator: AMD Instinct MI60 / MI50,
gfx906, 31.96 GiB free VRAM, ROCm HIP PASS. The exact Qwen3.8-27B-Q4_K_M
artifact is absent; available Qwen3-8B artifacts are rejected by the explicit
Qwen35 configuration guard. No substitute benchmark is valid for M26.

## Exact-model route A/B — 2026-09-16

Hardware state was valid: MI60/MI50 gfx906 at 1606/1000 MHz. The exact
Qwen3.8-27B-Q4_K_M model was used with the `m25_interactive` preset, P512,
TG128, five iterations, and `--no-stream`.

| Route | Median ms/token | tok/s |
| --- | ---: | ---: |
| Layer-major prefill + queued HIP graph decode | 60.5365 | 16.5189 |
| Layer-major prefill + graph-disabled `step()` decode | 83.2347 | 12.0142 |

Queued graph decode saves 22.6982 ms/token (27.3%) relative to the same
runtime with graphs disabled. This validates the route recovery change, but
the result remains above the M26 recovery gate of 33 ms/token. The remaining
gap is not attributed to kernels by this experiment.

**A/B decision:** KEEP the unified no-stream graph route; M26-B remains open.

## Qualified route rerun — 2026-09-16

With `Performance Level=manual`, SCLK `1606 MHz`, and MCLK `1000 MHz`, the
exact-model command was repeated at P512/TG128 for five iterations:

| Route | Median ms/token | tok/s |
| --- | ---: | ---: |
| Layer-major prefill + queued HIP graph decode | 59.8578 | 16.7063 |

This is the current qualified route result, but it remains above the M26
recovery gate of 33 ms/token. The idle post-run power reading is not used as a
qualification failure because the required manual clock levels were present
during the run; power-state telemetry should be captured during future runs.

## Corrected legacy comparison — 2026-09-16

The earlier attempted legacy comparison was invalid because `m25_interactive`
reapplies the layer-major preset after clearing overrides. Running without the
preset, with `MIINFER_PREFILL_LAYER_MAJOR=0`, produced a true legacy route
under the same manual 1606/1000 MHz clock state:

| Route | Median ms/token | tok/s |
| --- | ---: | ---: |
| Legacy prefill + queued HIP graph decode | 32.3865 | 30.877 |
| Interactive layer-major prefill + queued HIP graph decode | 59.8578 | 16.7063 |

The legacy route meets the M26 recovery gate. The interactive production route
does not; the remaining 27.4713 ms/token differential is attributable to the
layer-major execution contract and requires further isolation. This is not a
frontier or end-to-end production qualification result.

## Mx decode-flag isolation — REJECT, 2026-09-16

Removing `MIINFER_MX_MMV=1` and
`MIINFER_PREFILL_WIDE_MX_REPACKED_ATTN_DECODE=1` from the interactive preset
was tested at P512/TG128 for five iterations under manual 1606/1000 MHz
clocks. It degraded to `510.448 ms/token` (`1.959 tok/s`) and increased the
runtime allocation footprint. The flags are therefore required by the current
layer-major execution contract; this experiment does not identify them as the
source of the 27.4713 ms/token gap.

**Decision:** REJECT flag removal; restore the interactive preset unchanged.

## Mx MMV isolation — REJECT, 2026-09-16

Keeping `MIINFER_PREFILL_WIDE_MX_REPACKED_ATTN_DECODE=1` but removing
`MIINFER_MX_MMV=1` was tested at P512/TG128 for five iterations under manual
1606/1000 MHz clocks. It degraded to `529.903 ms/token` (`1.887 tok/s`). The
MMV flag is required by the current wide layer-major allocation/dispatch
contract. It was restored; no performance conclusion is drawn from the
contaminated candidate beyond rejecting this flag removal.

**Decision:** REJECT `MIINFER_MX_MMV` removal.

## Wide Mx attention-decode isolation — REJECT, 2026-09-16

Keeping `MIINFER_MX_MMV=1` but removing
`MIINFER_PREFILL_WIDE_MX_REPACKED_ATTN_DECODE=1` was tested at P512/TG128 for
five iterations under manual 1606/1000 MHz clocks. It degraded to
`130.566 ms/token` (`7.659 tok/s`). The flag was restored.

**Decision:** REJECT attention-decode flag removal; both Mx flags remain part
of the current wide layer-major contract.

## Profile-boundary check — 2026-09-16

The one-sample `MIINFER_DECODE_PROFILE=1` P512 run reported 60.058 ms/token,
but its operator events were inherited from the wide-prefill pass. The report
showed deferred-prefill-tail work as 40.8% of sampled events; it does not
profile the kernels executed inside the queued decode graphs. Therefore this
output is not a decode bottleneck attribution and must not be used to select a
kernel optimization.

**Decision:** RETEST — add graph-compatible decode instrumentation before
classifying the remaining 27.5 ms/token gap.

## Instrumented rerun — invalid, 2026-09-16

The first run after adding contract counters measured `117.466 ms/token` at
P512/TG128, compared with `60.5365 ms/token` for the prior valid run. The
post-run hardware query reported the GPU in a low-power state at 30 W, despite
the displayed clock targets. This run is contaminated and is not classified as
an instrumentation regression or an M26 performance result.

## Five-iteration decode-curve rerun — 2026-09-16

The exact interactive command was rerun with `--max-tokens 128` (the decode
curve requires 128 generated tokens) under manual 1606/1000 MHz clocks:

| Route | Context | Iterations | Median ms/token | tok/s |
| --- | ---: | ---: | ---: | ---: |
| Layer-major prefill + queued HIP graph decode | 512 | 5 | 59.2225 | 16.8855 |

This confirms the queued graph path is repeatable, but the production
layer-major route remains above the 33 ms/token M26 recovery gate.

## Native-buffer substitution — REJECT, 2026-09-16

An opt-in probe set `MIINFER_PREFILL_REPACKED_RESIDENT_ALL=0` while retaining
the wide layer-major prefill. A short eight-token check completed at 27.428
ms/token, but the sustained 128-token generation emitted only 15 tokens and
measured 345.185 ms/token. This fails the exact-generation requirement and is
not a valid performance candidate.

**Decision:** REJECT. Native buffers cannot be selected independently of the
wide-prefill state/layout contract; the probe switch was removed.

## Mx single-token MMV selector isolation — diagnostic, 2026-09-19

Added `MIINFER_PRESET=m26_mx_mmq` as a reproducible control vector. It is
identical to `m25_interactive` except that it leaves `MIINFER_MX_MMV` unset;
resident weights, Mx attention decode, graph settings, prompt, context, and
generation length are unchanged. Runs used the exact model hash recorded
above, manual SCLK/MCLK 1606/1000 MHz, P512, TG128, five curve iterations.
The four runs were interleaved A/B/A/B:

| Arm | Preset | Median ms/token | tok/s |
| --- | --- | ---: | ---: |
| A | `m26_mx_mmq` (MMQ control) | 544.431 | 1.8368 |
| B | `m25_interactive` (MMV) | 72.2632 | 13.8383 |
| A | `m26_mx_mmq` (MMQ control) | 528.178 | 1.8933 |
| B | `m25_interactive` (MMV) | 56.8187 | 17.5998 |

The two MMQ medians average 536.3045 ms/token; the two MMV medians average
64.5410 ms/token, an 8.31x selector-level speedup. Both arms completed all
128 tokens in each of five iterations. Clock and power spot-checks during the
runs showed SCLK/MCLK 1606/1000 MHz and approximately 126 W under load.

An eight-token direct-generation smoke emitted different text between the two
arms (`7,168 stream processors` vs. `56 compute units (CUs`). Therefore this
measurement establishes a large execution-cost difference, not token parity
or numerical correctness for the full layer-major route. Existing EXP-0335
correctness evidence remains scoped to its own P512 continuation checks.

**Decision:** KEEP MMV in the interactive route; reject MMQ as the decode
recovery path. Neither arm meets the M26 `<=33 ms/token` gate, and the
MMV-versus-MMQ output difference requires a numerical/logit oracle before any
correctness claim. This selector isolation does not close the historical
M26-B differential gate.

## Synchronous decode operator profile — 2026-09-19

To avoid the queued-graph profiler boundary, the existing stage profiler was
run with HIP graphs disabled and the MMV production selector enabled. This
was one `--decode-curve` iteration at context 512, generating 128 tokens, with
the sampled token at absolute position 512. The run completed at
`60.7138 ms/token` (`7771.36 ms` decode total). Profiled GPU stage time for
the selected token was `61.2578 ms`; the aggregate layer-loop timer was
`65.2396 ms` and is gross timing, not additive to the stage family totals.

```bash
MIINFER_PRESET=m25_interactive MIINFER_HIP_GRAPH=0 \
MIINFER_DECODE_PROFILE=1 MIINFER_DECODE_PROFILE_POSITION=512 \
build/mi50-release/miinfer run \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  --context 16384 --decode-curve --curve-context 512 \
  --curve-iterations 1 --no-stream --max-tokens 128
```

| Sampled family | GPU ms | Share |
| --- | ---: | ---: |
| projection or norm | 14.0758 | 22.98% |
| FFN gate/up projection | 11.3611 | 18.55% |
| KV or head normalization | 11.2577 | 18.38% |
| FFN down projection | 5.8165 | 9.50% |
| projection or attention output | 4.2773 | 6.98% |
| SwiGLU | 4.1573 | 6.79% |
| residual | 2.6963 | 4.40% |
| projection or RoPE | 2.6886 | 4.39% |
| residual/post norm | 1.4254 | 2.33% |
| projection or K | 1.2197 | 1.99% |

The stage sum nearly matches the measured per-token decode latency. This
supports classifying the current route as GPU-operator dominated rather than
host synchronization dominated. These broad profiler families do not yet
identify resolved kernel choices or explain the historical 33 ms/token route;
the M26-B gate remains open.

## Default production runtime recovery screen — 2026-09-19

With no `MIINFER_*` variables inherited and no preset selected, the ordinary
CLI/server `Qwen35RuntimeEngine` route ran the built-in curve at P512/TG128.
Hardware was in manual mode at SCLK/MCLK 1606/1000 MHz; in-run telemetry
showed the same clocks under load. The exact model hash is recorded in the
audit above. Five iterations each completed all 128 generated tokens:

| Route | Median decode ms/token | tok/s | Result |
| --- | ---: | ---: | --- |
| Default runtime (legacy prefill + queued graph decode) | 32.2948 | 30.9647 | <=33 gate PASS |

The full `mi50-release` CTest suite then passed 24/24 tests, including the GPU
decode-sequence and forward tests. The measured default runtime route
therefore meets the M26 recovery latency gate with current-tree correctness
tests passing. This does not qualify `m25_hi_qualified` or the experimental
wide/Mx route, and does not close the separate M26-B historical route
attribution. No sub-30 ms optimization is authorized by this result.
