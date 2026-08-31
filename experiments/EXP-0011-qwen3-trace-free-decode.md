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

The code is frozen at clean commit `dd275325cfb9`; both captures report
`git_dirty=false` and the pinned model SHA256
`458634762bea7dbe19f3ce0614465bafd15ee90e815229c427043afcf195d628`.

The apples-to-apples control is
`bench/results/20260831T200125Z-337651/`: one prompt token, no generated
warmup forwards, and seven measured forwards—the same seven post-first-token
forwards used by M5-A.

| Metric | Mean | Median | Throughput |
|---|---:|---:|---:|
| Sequential prompt ingestion | 29.368 ms | 28.771 ms | 34.051 tok/s |
| TTFT | 29.642 ms | 29.044 ms | — |
| Measured decode (7 forward calls) | 222.163 ms | 222.082 ms | **31.508 tok/s** |

Relative to M5-A's `22.658 tok/s`, this is a `1.391×` throughput increase,
or `28.09%` lower measured decode-forward time. The generated IDs remain
`[8,341,286,470,330,9707,11,330]`.

The longer steady-state capture is
`bench/results/20260831T200016Z-336367/`: eight warmup generated tokens and
64 measured forwards. It reports `5030.041 ms` mean decode time, or
`12.724 tok/s`, as context grows from 8 to 72 tokens. Its result is not an
apples-to-apples replacement for the short M5-A baseline, but it is useful for
future context-scaling comparisons.

## Interpretation

This is a measurement control, not a production optimization decision. Any
future candidate must be compared against this trace-free control and retain
the correctness checks. M5-A remains the historical correctness-first
baseline, while this control is the appropriate performance reference for
changes to the serving path.

## Decision

`KEEP` — the trace-free control is deterministic, preserves the pinned
sequence, and removes diagnostic trace copies from the measured path. It is
the correct control for the first optimization A/B; M5-A remains unchanged as
the historical correctness-first baseline.

## Follow-up

Use the trace-free result to quantify the cost of diagnostic work. Then test
one measured optimization hypothesis, likely dispatch/materialization removal
or FFN projection, with an interleaved A/B benchmark.
