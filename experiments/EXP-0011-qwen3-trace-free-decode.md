# EXP-0011 — Trace-free Qwen3-8B decode control

## Hypothesis

The M5-A number is materially affected by diagnostic trace copies and per-call
trace construction. A trace-free benchmark can measure the same decode
semantics with negligible measurement-specific work and provide the control
for subsequent optimizations.

## Motivation

M5-B attributed work using intrusive per-operation HIP events, but that profile
cannot be compared directly with the 22.658 tok/s M5-A result. This slice
removes trace collection from the benchmark path without changing the normal
correctness executor or kernel behavior.

## Baseline

M5-A Release baseline, commit `2054995`, measured seven post-first-token
forward calls at `22.658 tok/s` and `39.061 ms` TTFT for the one-token `hello`
prompt. Model SHA256:
`458634762bea7dbe19f3ce0614465bafd15ee90e815229c427043afcf195d628`.

## Candidate

Commit under test adds `execute_qwen3_decode_gpu_fast()`. It uses the existing
embedding, 36-layer execution, KV cache, final norm, and LM-head operations,
but suppresses all layer checkpoint captures and copies only final logits to a
preallocated host buffer for greedy selection. The benchmark warms eight
generated tokens and measures 64 further decode forward calls.

## Environment

Target: AMD Instinct MI50 / gfx906, Qwen3-8B Q4_0. The runner stores hardware
telemetry, before/after environment snapshots, model hash, and raw timing
samples in the result directory.

## Benchmark

```bash
scripts/run-m5c0-fast-decode.sh /path/to/Qwen3-8B-q4_0-b968826d.gguf
```

Equivalent direct options are `--warmup 8 --generated-tokens 64
--iterations 5`. The measured decode interval contains 64 forward calls and
CPU argmax selection; prompt ingestion and warmup are excluded from that
interval.

## Correctness

Every run requires deterministic generated IDs and finite logits. For the
default prompt `[14990]`, the first eight generated IDs must remain
`[8,341,286,470,330,9707,11,330]`. The fast path performs no trace copies.

## Results

To be filled from the first clean-commit capture.

| Metric | Mean | Median | Throughput |
|---|---:|---:|---:|
| Sequential prompt ingestion | — | — | — |
| TTFT | — | — | — |
| Measured decode (64 forward calls) | — | — | — |

## Interpretation

This is a measurement control, not a production optimization decision. Any
future candidate must be compared against this trace-free control and retain
the correctness checks. M5-A remains the historical correctness-first
baseline, while this control is the appropriate performance reference for
changes to the serving path.

## Decision

`PENDING` — capture clean results before selecting M5-C1.

## Follow-up

Use the trace-free result to quantify the cost of diagnostic work. Then test
one measured optimization hypothesis, likely dispatch/materialization removal
or FFN projection, with an interleaved A/B benchmark.
