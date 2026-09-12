# EXP-0326 — M25-D GDN four-column wave shard

## Hypothesis

The pinned gfx906 GDN kernel assigns four state columns to each Wave64. MIInfer
uses two columns per wave in its otherwise state-correct Mx scan. Matching the
larger register shard should reduce the number of state-column blocks without
changing the public `[head][key][value]` state contract.

## Baseline

`mx_gdn_chunk_kernel<128, 64, 2>` selected by the qualified M25 vector when
`MIINFER_MX_GDN_TC4` is unset or `0`.

## Candidate

`mx_gdn_chunk_kernel<128, 64, 4>` selected by the temporary strict
`MIINFER_MX_GDN_TC4=1` flag. The launch grid changes from 32 to 16 state-column
blocks per value head, matching the pinned `Tc=4`/two-Wave64 mapping. All
weights, inputs, arithmetic, state orientation, and recurrent orchestration
remain unchanged.

## Environment

- AMD Instinct MI50 / gfx906 / Wave64
- Qwen3.8-27B-Q4_K_M GGUF
- model SHA-256 `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- ROCm `7.1.52802-9999`, clang `20.0.0.rocm`
- release build, context capacity `1024`
- post-run clocks `1606/1000 MHz`, temperatures `33/34/33 C`

## Benchmark

Standalone serial A/B used `build/mi50-release/miinfer-m12-gdn-chunk-bench`.
The production A/B used the explicit hermetic M25 H/I vector, `--max-tokens 0`,
and a 512-token prompt. Three exploratory pairs used the 512-token `hello`
shape; one final pair used the exact repeated-fox prompt from EXP-0323.

## Correctness

The standalone candidate passed its CPU recurrent oracle with:

```text
mx_max_output_error=0.000000011
mx_max_state_error=0.000000082
```

The exact-fox candidate also passed `--repeat-p512-check` in one process:
first token `13477` (`brown`), continuation token `37550`, first prefill
`2489.73 ms`, and repeat prefill `2426.00 ms`.

## Results

Standalone serial medians from three control/candidate samples:

| path | median us |
|---|---:|
| Mx GDN Tc2 | 1,029.917 |
| Mx GDN Tc4 | 893.119 |

This is a `13.3%` isolated GDN improvement. The first two production hello
pairs were:

| path | P512 ms | tok/s |
|---|---:|---:|
| control | 2505.66, 2505.90 | 204.34, 204.32 |
| candidate | 2369.10, 2369.88 | 216.12, 216.04 |

A third pair moved together at `2535.65 ms` control and `2543.99 ms`
candidate, so it is retained as machine-state spread rather than a conclusion.
The exact-fox pair measured `2523.28 ms / 202.91 tok/s` control versus
`2419.68 ms / 211.60 tok/s` candidate. Allocation was identical at
`21,993,242,964 B`. A further exact-fox pair with continuous telemetry held
SCLK/MCLK at `1606/1000 MHz`: `2445.18 ms / 209.39 tok/s` control versus
`2372.06 ms / 215.85 tok/s` candidate, again with identical allocation.

## Interpretation

The larger state shard is numerically safe and materially faster in the
standalone kernel. The production samples support an approximately `4–6%`
P512 improvement, but they are not yet continuous-telemetry qualification.
The single continuous-telemetry pair validates the result but is not yet a full
interleaved qualification series. The flag therefore remains opt-in; the
versioned H/I preset stays on Tc2 until that series is collected.

## Decision

**KEEP opt-in; RETEST for default promotion.**

## Follow-up

Run an interleaved exact-fox A/B with continuous SCLK/MCLK telemetry, then
re-run the long-generation and repeated-P512 gates before adding Tc4 to the
qualified preset.
