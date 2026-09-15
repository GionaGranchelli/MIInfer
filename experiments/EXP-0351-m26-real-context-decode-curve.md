# EXP-0351 — M26 real-context decode curve

Status: RETEST — reconnaissance only; no optimization decision.

## Hypothesis

The Qwen3.8-27B serving decode path loses throughput as existing context
grows, primarily through context-dependent attention/KV work.

## Benchmark

The `miinfer run --decode-curve` mode prepares a fresh prompt at each context,
then measures 128 generated tokens. It reports decode-only wall time and
throughput. The production `m25_interactive` flags were used with a 4096
position HIP-graph capture limit.

## Initial reconnaissance

MI50/gfx906, Qwen3.8-27B-Q4_K_M, context capacity 16384, one sample per point.
The 16384 point was skipped because the benchmark reserves 128 output tokens.

| Existing context | Decode ms / 128 | ms/token | tok/s |
| ---: | ---: | ---: | ---: |
| 512 | 8267.95 | 64.593 | 15.48 |
| 2048 | 8420.44 | 65.785 | 15.20 |
| 4096 | 10266.4 | 80.206 | 12.47 |
| 8192 | 7821.55 | 61.106 | 16.37 |
| 12288 | 8116.29 | 63.409 | 15.77 |

## Interpretation

The single-pass curve is internally non-monotonic and therefore contaminated
or insufficiently sampled for attribution. It does not prove or reject the
attention/KV hypothesis. No kernel change is accepted from this result.

## Follow-up

Repeat with at least five interleaved samples, context capacity above 16K, and
hardware-state capture. Add per-operator GPU timing and dispatch/synchronization
counts before choosing an optimization target.
