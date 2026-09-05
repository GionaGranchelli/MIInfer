# EXP-0122 — M6-B30 transposed DeltaNet recurrent state

## Question

Can a gfx906-friendly transposed recurrent-state layout preserve Qwen3.8
logical state behavior while reducing DeltaNet state-update time?

## Hypothesis

The existing logical `[value_head][row][column]` state makes each Wave64 lane
walk strided memory. A physical `[value_head][column][row]` layout gives each
lane contiguous row access without changing the logical state or recurrence.

## Baseline and candidate

The control is the existing no-decay-store state update and row-major physical
state layout, selected with `MIINFER_DELTA_TRANSPOSED_STATE=0`. The candidate
uses the same recurrence and arithmetic with a physical column-major state
layout; logical upload/download and diagnostic fingerprints transpose at the
boundary. The candidate is now the default.

## Environment

```text
GPU:       AMD Instinct MI50 / gfx906
Clock:     stable_peak (1725 MHz SCLK / 1000 MHz MCLK observed)
Model:     /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Build:     build/mi50-release
Fixture:   /tmp/m6a273-reference-p12
```

## Commands

```bash
MIINFER_DELTA_TRANSPOSED_STATE=0 build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block MODEL FIXTURE --bench64
MIINFER_DELTA_TRANSPOSED_STATE=1 build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block MODEL FIXTURE --bench64
MIINFER_DELTA_TRANSPOSED_STATE=0 build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block MODEL FIXTURE --bench128
MIINFER_DELTA_TRANSPOSED_STATE=1 build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block MODEL FIXTURE --bench128
MIINFER_DELTA_TRANSPOSED_STATE=1 build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block MODEL FIXTURE --generate64
MIINFER_DELTA_TRANSPOSED_STATE=1 build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block MODEL FIXTURE --generate128
MIINFER_DELTA_TRANSPOSED_STATE=1 build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block MODEL /tmp/m6a273-reference-p12 --prefix64-observable-contract
```

## Correctness

The isolated state-update candidate matches the control's logical output and
state at positions 0 and 1 within the existing tolerances. The representative
microbenchmark reduced state-update time from `202.979 us` to `118.153 us`
(`41.79%`).

Native generation passed deterministic replay at 16, 64, and 128 tokens. The
64-token and 128-token runs reported first token `11`, last tokens `271` and
`310`, respectively, zero decode-loop allocations, and `17,019,965,780`
tracked/peak device bytes.

The complete `/tmp/m6a273-reference-p12` observable run completed through P64
under the accepted A27 margin-aware external contract. Final P64 logits had
cosine `0.999633`, top-5 overlap `5/5`, and matching argmax `8719`. Known late
internal recurrent-state envelope warnings remain diagnostic only. The
shorter `/tmp/m6a273-reference` fixture is incomplete at position 12 and is
not used as the acceptance fixture.

Release CTest passed `20/20`.

## Benchmark

Five-sample medians per process at stable_peak:

| Workload | Control median | Candidate median | Change |
| --- | ---: | ---: | ---: |
| TG64 ms | 5580.80 | 5396.86 | -3.30% |
| TG64 tok/s | 11.4679 | 11.8587 | +3.41% |
| TG128 ms | 11320.70 | 10925.30 | -3.49% |
| TG128 tok/s | 11.3067 | 11.7159 | +3.62% |

Both candidate workloads replayed exactly. Device usage was unchanged and no
decode allocation occurred.

## Interpretation

This changes only physical state layout and the required diagnostic transpose;
the recurrent math, logical cache/state contract, and execution ordering are
unchanged. The microbenchmark and both whole-token workloads show a repeatable
gain, with no VRAM cost.

## Decision

**KEEP; production-selected.** The transposed recurrent-state layout is the
default. `MIINFER_DELTA_TRANSPOSED_STATE=0` retains the row-major control for
future A/B comparisons.

## Follow-up

Run the post-selection profile and compare the remaining recurrent stages
before choosing the next optimization.
