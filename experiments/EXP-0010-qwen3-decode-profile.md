# EXP-0010 — Qwen3-8B steady-state decode profile

## Hypothesis

The approximately 44 ms per-token M5-A decode baseline can be decomposed into
operation-family GPU work, device copies, and dispatch overhead well enough to
select the first M5 optimization target.

## Motivation

M5-A established the immutable correctness-first baseline. This experiment is
profiling-only: it changes no default execution semantics and deliberately
does not claim optimized throughput.

## Baseline

M5-A Release baseline at commit `2054995`:

* Qwen3-8B Q4_0, model SHA256
  `458634762bea7dbe19f3ce0614465bafd15ee90e815229c427043afcf195d628`;
* one-token `hello` prompt (`14990`), eight greedy generated IDs;
* decode mean `308.944 ms` for seven forward calls, or `22.658 tok/s`;
* observed peak VRAM `4.636 GiB`.

## Candidate

An opt-in `Qwen3GpuProfile` records HIP-event time around each operation
launch, synchronous device-copy time, and dispatch counts for one warmed
position-1 decode. The fixed warmup token is `14990`; the profiled token is
`8`. The profile path retains existing trace copies and allocations.

## Environment

Target: AMD Instinct MI50 / gfx906, Qwen3-8B Q4_0. The exact model hash and
machine state are recorded by `scripts/run-bench.sh` in the retained result
directory. External ROCm profilers were not available in the benchmark
environment, so this is an internal HIP-event profile.

## Benchmark

```bash
scripts/run-m5b-profile.sh /path/to/Qwen3-8B-q4_0-b968826d.gguf
```

The profile is one steady-state decode token after an unprofiled warmup. Its
wall time includes per-operation synchronization and diagnostic host copies;
category GPU times are the intended attribution data.

## Correctness

The profile run requires finite logits and cache length 2 after the profiled
token. Normal execution receives a null profile pointer and follows the
existing launch path.

## Results

To be filled from the first clean-commit capture:

| Operation family | GPU ms | Copy ms | Dispatches | Share of GPU time |
|---|---:|---:|---:|---:|
| embedding | — | — | — | — |
| normalization | — | — | — | — |
| quantization | — | — | — | — |
| Q/K/V projection | — | — | — | — |
| O projection | — | — | — | — |
| FFN projection | — | — | — | — |
| RoPE | — | — | — | — |
| attention | — | — | — | — |
| activation | — | — | — | — |
| residual | — | — | — | — |
| conversion | — | — | — | — |
| LM head | — | — | — | — |

The exact JSON and telemetry directory will be linked here after capture.

## Interpretation

No optimization decision is made by this experiment. The wall time must not
be compared with M5-A because profiling serializes operation attribution and
the current executor copies diagnostic traces. The first optimization slice
must benchmark an unprofiled candidate against the M5-A workload and preserve
the correctness gates.

## Decision

`PENDING` — capture on a clean commit, then select one measured bottleneck.

## Follow-up

Use the largest attributable operation family or synchronization/dispatch
component as the single hypothesis for M5-C. Record any negative result as a
separate experiment outcome rather than rewriting this record.
