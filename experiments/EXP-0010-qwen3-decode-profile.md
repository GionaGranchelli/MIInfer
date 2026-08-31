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

Canonical clean-commit capture:
`bench/results/20260831T194351Z-333304/`

The result records commit `a152d6f2340a`, `git_dirty=false`, and the pinned
model SHA256
`458634762bea7dbe19f3ce0614465bafd15ee90e815229c427043afcf195d628`.

| Operation family | GPU ms | Copy ms | Dispatches | Share of GPU time |
|---|---:|---:|---:|---:|
| embedding | 0.008160 | 0.000000 | 1 | 0.03% |
| normalization | 2.917432 | 0.000000 | 289 | 11.01% |
| quantization | 3.933754 | 0.000000 | 505 | 14.84% |
| Q/K/V projection | 1.703200 | 0.000000 | 108 | 6.43% |
| O projection | 0.814559 | 0.000000 | 36 | 3.07% |
| FFN projection | 7.035994 | 0.000000 | 108 | 26.55% |
| RoPE | 0.583677 | 0.000000 | 72 | 2.20% |
| attention | 3.461915 | 0.000000 | 36 | 13.06% |
| activation | 0.296319 | 0.000000 | 36 | 1.12% |
| residual | 0.506880 | 0.000000 | 72 | 1.91% |
| conversion | 2.302078 | 0.000000 | 324 | 8.69% |
| LM head | 2.935037 | 0.000000 | 1 | 11.08% |

Totals were `26.499005 ms` of GPU event time, `19.547981 ms` of device-copy
time, and `1,588` profiled GPU dispatches. End-to-end profiled wall time was
`88.035089 ms`, leaving `41.988103 ms` unaccounted for event creation,
synchronization, allocation, and other instrumentation/executor overhead.
The reported wall time is therefore intentionally not a throughput number.

Peak sampled VRAM was `4.634 GiB` (`4,975,685,632` bytes). Telemetry during
the deliberately serialized profile observed changing clocks, so this run is
not a valid clock-controlled performance baseline; the environment files are
retained for that reason.

## Interpretation

No optimization decision is made by this experiment. The wall time must not
be compared with M5-A because profiling serializes operation attribution and
the current executor copies diagnostic traces. The first optimization slice
must benchmark an unprofiled candidate against the M5-A workload and preserve
the correctness gates.

## Decision

`KEEP` — the profile provides the requested M5-B attribution without changing
default execution semantics. The largest measured GPU family is FFN
projection (`7.036 ms`, `26.55%` of summed GPU event time), followed by
quantization (`3.934 ms`) and attention (`3.462 ms`). The 1,588 dispatches and
large trace-copy/instrumentation overhead also make dispatch reduction and a
trace-free serving path important candidates, but neither is accepted as an
optimization without an unprofiled A/B measurement.

## Follow-up

Use one of the measured bottlenecks as the single M5-C hypothesis, with an
unprofiled A/B benchmark against the M5-A workload. Preserve this profile and
record any negative result as a separate experiment outcome rather than
rewriting this record.
