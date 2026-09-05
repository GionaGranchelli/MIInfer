# EXP-0127 — M6-B35 Q4_K×Q8_1 LDS activation reuse

## Question

Can recurrent and full-attention Q4_K×Q8_1 FFN projections reuse one
LDS-resident activation tile across two independent output rows while keeping
the existing per-row reduction and arithmetic?

## Hypothesis

The current 128-thread MMVQ workgroup rereads the same Q8_1 activation blocks
for every output row. A 256-thread workgroup can stage the activation tile once
in LDS and run two unchanged 128-thread row reductions against it.

## Baseline and candidate

The baseline is B32 at commit `6ee887e`, using one 128-thread workgroup per
output row. The candidate uses two independent 128-thread row groups in one
256-thread workgroup and an LDS-resident Q8_1 activation tile. Quantization,
weights, dot-product arithmetic, and each row's reduction order are unchanged.
The candidate is selected by default; `MIINFER_Q4K_Q8_1_LDS_INPUT=0` selects
the baseline.

## Environment

```text
GPU:       AMD Instinct MI50 / gfx906
Clock:     stable_peak (1725 MHz SCLK / 1000 MHz MCLK observed)
Model:     /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Build:     build/mi50-release
Fixture:   /tmp/m6a273-reference-p12
Baseline:  6ee887e
```

## Correctness

Native generation passed deterministic replay at 16, 64, and 128 tokens. The
64-token run ended at token `28609`; the 128-token run ended at `2791`. Decode
allocations were `0` and tracked/peak device bytes remained
`17,019,965,780`.

The complete P64 external observable contract passed without tolerance
changes. Teacher-forced decisions matched in the tested run; P64 argmax was
`8719`, logits cosine was `0.999603`, and top-5 overlap was `4/5`, under the
accepted margin-aware contract. Release CTest passed `20/20`.

## Benchmark

Five-sample medians per process at stable_peak:

| Workload | B32 control median | Candidate median | Change |
| --- | ---: | ---: | ---: |
| TG64 ms | 5305.16 | 5151.54 | -2.90% |
| TG64 tok/s | 12.0637 | 12.4235 | +2.98% |
| TG128 ms | 10757.8 | 10443.6 | -2.92% |
| TG128 tok/s | 11.8983 | 12.2563 | +3.01% |

The candidate keeps the same logical dispatch count. Its dynamic LDS tile is
at most `19,584` bytes for the 17,408-element FFN dimension; persistent and
peak tracked device usage did not change.

## Interpretation

The candidate removes repeated activation reads at the workgroup level while
retaining the existing per-row computation. It produces a repeatable roughly
3% whole-token improvement at both generation lengths, with no observed
correctness regression or memory-budget cost.

## Decision

**KEEP; production-selected.** The LDS activation-reuse path is the default.
`MIINFER_Q4K_Q8_1_LDS_INPUT=0` retains the B32 control for A/B comparisons.

## Follow-up

Refresh the post-B35 profile before selecting the next optimization. Do not
repeat the rejected two-row reduction geometry or fused SiLU experiment without
a materially different mechanism.
