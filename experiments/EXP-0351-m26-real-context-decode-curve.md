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

## Follow-up

Run the single-context decode attribution mode at 12288 with graphs disabled,
then classify fixed versus context-dependent stage costs and count dispatches
and synchronizations before choosing an optimization target.
