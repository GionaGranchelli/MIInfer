# EXP-0221 — M11-B Direct Consumption of Batched Prefill Workspace

## Hypothesis

Layer-major prefill already computes normalized activations and Q/K/V
projections into persistent B=4 workspaces, but `run()` copied them back into
per-token scratch buffers before consuming them. Passing those workspace
pointers directly should remove device-to-device copies without changing
causal ordering or kernel math.

## Candidate

Recurrent layers directly consume prepared normalized, QKV, and attention-gate
buffers. Full-attention layers directly consume prepared normalized, Q/K, and
V buffers. Scalar and token-major paths retain their original storage.

## Environment

- GPU: AMD Instinct MI60/MI50-visible gfx906
- SCLK/MCLK: 1606/1000 MHz
- ROCm: 6.4.0 / LLVM 20
- Build: `mi50-release`
- Model: `Qwen3.8-27B-Q4_K_M.gguf`
- Candidate: layer-major, native Q4/Q5/Q6 projections, deferred tails

## Correctness

P17 and P128 produced the same layer-major continuation as the committed
candidate control. Full CTest passed all 21 tests. P128 decode measured
`33.29 tok/s`.

## Results

Three P513 production-shaped repetitions:

| Path | Runs (ms) | Median | PP tok/s |
| --- | --- | ---: | ---: |
| Committed candidate | 11386.71 / 11395.72 / 11396.37 | 11395.72 | 45.02 |
| Direct workspace consumption | 11183.27 / 11323.54 / 11259.36 | 11259.36 | 45.56 |

The copy-elimination path improves the measured median by approximately
`1.2%` and adds no workspace allocation. A supporting P128 run measured
`45.28 tok/s` prefill.

## Decision

**KEEP as an opt-in layer-major cleanup.** It is correct and provides a small
production-shaped gain, but it leaves the M11-B gate far away at 45.56 PP
tok/s. The default token-major path remains unchanged.

## Follow-up

The remaining gap still requires a different quantized GEMM/dataflow strategy;
copy elimination is not sufficient.
