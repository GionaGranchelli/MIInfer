# EXP-0351 — M26 real-context decode floor and context penalty

Status: ACTIVE — context attribution complete; fixed-floor regression audit next.

## Revised question

Why is the fixed decode floor approximately `59 ms/token` in the current
serving path when earlier MIInfer work qualified approximately `33 ms/token`,
and which fixed-cost families can recover the missing `18–24 ms/token`?

M26 has two independent objectives: reduce the 512-context floor toward
`35–37 ms/token`, and reduce the 12K context penalty to `<=2 ms/token`. The
primary gate remains `<=40 ms/token` at 12–16K (`>=25 tok/s`).

## Revised gates

| Gate | Requirement |
| --- | --- |
| M26-A | 12K context penalty `<=2 ms/token` |
| M26-B | 512 fixed floor `<=50 ms/token` (`>=20 tok/s`) |
| M26-C | 512 fixed floor `<=44 ms/token` (`>=22.7 tok/s`) |
| M26-D | 512 fixed floor `<=37–38 ms/token` |
| M26-E | 12K wall time `<=40 ms/token` (`>=25 tok/s`) |

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

The fixed floor is now the dominant M26 problem: removing the context penalty
alone would still leave the runtime near 17 tok/s. Attention is the measured
context-scaling bottleneck, not the general decode bottleneck.

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

The same diagnostic was then run at `position=12288` with the same runtime
vector and graph state. It reported `64.179 ms` of accounted operator timing,
`68.009 ms` gross whole-layer timing, and `63.669 ms` wall ms/token. Relative
to 512, the accounted total increased by `5.390 ms/token`. The largest family
delta was projection or attention output (`4.104 -> 10.457 ms`, `+6.353 ms`);
projection/norm (`13.420 -> 13.277 ms`), FFN Gate/Up (`10.842 -> 10.705 ms`),
KV/head norm (`10.732 -> 10.695 ms`), and FFN Down (`5.735 -> 5.653 ms`)
were effectively constant. This is evidence that context-dependent work is
localized to the full-attention output/KV path, while the recurrent fixed
floor remains the primary short-context cost.

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

Audit the historical M8/M9 `~33 ms/token` result against the current M25
serving path with identical model, clocks, context, graph state, generation
length, and sampling. Break the fixed floor into recurrent/attention layers,
QKV/GDN/state update, FFN, normalization, conversion/quantization, LM head,
sampling, and graph/runtime overhead. Rank by absolute ms/token and pursue
only candidates with a credible `>=1 ms/token` end-to-end saving.

Keep the attention output/KV investigation capped at its expected `4–6
ms/token` recovery. Do not start another generic FFN experiment until the
approximately `24.5 ms/token` historical difference is explained.
