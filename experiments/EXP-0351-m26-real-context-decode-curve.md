# EXP-0351 — M26 real-context decode curve

Status: RETEST — baseline curve complete; attribution pending.

## Hypothesis

The Qwen3.8-27B serving decode path loses throughput as existing context
grows, primarily through context-dependent attention/KV work.

## Benchmark

The `miinfer run --decode-curve` mode prepares a fresh prompt at each context,
then measures 128 generated tokens. It reports decode-only wall time and
throughput. The production `m25_interactive` flags were used with a 4096
position HIP-graph capture limit.

## Five-run baseline

MI50/gfx906, Qwen3.8-27B-Q4_K_M, context capacity 32768, five fresh samples
per point, fixed 128-token generation, and `m25_interactive` flags. GPU clocks
were allowed to run at the installed performance state. The benchmark reserves
128 output tokens, so capacity was set above the 16K point.

| Existing context | Decode ms / 128 | ms/token | tok/s |
| ---: | ---: | ---: | ---: |
| 512 | 7360.14 | 57.501 | 17.39 |
| 2048 | 7413.98 | 57.922 | 17.26 |
| 4096 | 7753.75 | 60.576 | 16.51 |
| 8192 | 7714.80 | 60.272 | 16.59 |
| 12288 | 8106.56 | 63.333 | 15.79 |
| 16384 | 7973.68 | 62.294 | 16.05 |

## Interpretation

The five-run median curve is relatively flat: 57.50 ms/token at 512 versus
62.29 ms/token at 16K, an increase of 4.79 ms/token. This rejects the claim
that context-scaled attention alone explains the missing 25 tok/s. It does not
yet provide the required operator attribution, because the current curve mode
reports only aggregate decode timing.

## Attribution decision boundary

The corrected decode profiler was validated at the first post-prompt token
(`position=512`) with graphs disabled. It reported `58.789 ms` of summed
operator timing and `62.627 ms` of gross whole-layer timing for a token whose
decode wall measurement was `59.124 ms`. The component accounting therefore
reconciles at the operator level; the gross layer interval is intentionally
not summed with its component events.

At this short-context point, the largest fixed-cost families were projection
or norm (`13.420 ms`), FFN Gate/Up (`10.842 ms`), KV or head norm
(`10.732 ms`), FFN Down (`5.735 ms`), and projection or attention output
(`4.104 ms`). The causal attention work is not the dominant fixed cost.

The existing qualified M24-E whole-tail measurement provides the current
best operator evidence: recurrent deferred execution accounted for
`3424.200 ms` across 48 layers, versus `905.333 ms` for the 16 attention
deferred tails. Its largest families were recurrent FFN Gate/Up plus SwiGLU
(`1493.790 ms`), FFN Down plus residual (`793.615 ms`), and GDN core
(`864.534 ms`). This is consistent with the M26 curve's large fixed floor and
does not support making KV traffic the first optimization target.

This is not yet the required decode-only 90% attribution: M24-E measured the
qualified wide-tail execution path, not 128 steady-state decode tokens. The
decode boundary profiler therefore remains required before accepting a kernel
change. Until then, the working hypothesis is fixed recurrent
materialization/dispatch cost, with attention treated as a secondary
context-dependent component.

## Follow-up

Validate the decode boundary profiler at one sampled token, then compare
recurrent and attention whole-layer timings at 512 and 12288. Count dispatches,
synchronizations, and allocations per token. Optimize recurrent deferred
materialization only after those decode measurements reconcile with wall time.
