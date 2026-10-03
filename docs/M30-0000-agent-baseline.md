# M30-0000 — Canonical Agent Baseline

`M30_AGENT_WORKLOAD_VERSION=m30-agent-v1`

## Provenance

| Item | Value |
|---|---|
| `M30_0000_BASE_SHA` | `23deab2eda689fc0cfc21782d3f6df1303bee2ee` |
| host | `fedora-workstation@100.118.66.80` (Z840, `HIP_VISIBLE_DEVICES=0`) |
| model | `Qwen3.8-27B-Q4_K_M.gguf` |
| context limit | 131072 |
| workload | 40 deterministic logical turns, 86 production requests |
| raw workload | `results/m30-0000-agent-workload.json` |
| corrected attribution | `results/m30-0000-agent-workload-corrected.json` |
| raw server log | `results/m30-0000-server-128k.log` |

The raw server log is authoritative for per-request prompt, prefix, suffix,
prefill, TTFT and total-request measurements. The first run's embedded
`server_latency` fields were stale repeated snapshots; the corrected JSON was
generated offline by remapping the 86 raw latency events in request order.
No inference was rerun.

## Session totals

| Metric | Result |
|---|---:|
| agent turns | 40 |
| requests | 86 |
| final context size | 82,087 tokens |
| total session wall-clock | 12,918,928 ms (3h 35m 18.928s) |
| total logical prompt tokens | 3,651,780 |
| total physically processed prompt tokens | 1,582,245 |
| repeated-prefix tokens | 2,069,535 |
| potential suffix-only tokens | 1,582,245 |
| repeated-work fraction | 56.67% |
| total generated tokens | 41,114 |
| peak context | 82,087 tokens |
| peak VRAM | 34,323,197,952 B |
| cold requests / reuse hits | 33 / 53 |

The session wall-clock is the sum of the sequential per-turn client request
wall times. The repeated-work fraction is an opportunity estimate, not a
claimed wall-clock speedup.

## Context regimes and micro-metrics

The natural run crossed approximately 8K, 32K and 64K and reached 82K; no
turn was enlarged solely to hit a checkpoint. Across all requests, raw-log
prefill time sums to 10,810,662 ms and raw-log TTFT-wall sums to 10,820,093
ms. Per-turn records retain TTFT, prefill, decode rate, total request time,
generated tokens, context and VRAM before/after.

The required generation gate passed for all 40 turns. The auxiliary
visible-answer gate was false for nine turns because the deterministic tool
flow produced short or empty final `content` fields; this is reported as a
quality boundary and is not substituted for the raw generation/latency
evidence.

## Status

`M30_AGENT_BASELINE_QUALIFIED`

No reuse mechanism was implemented.
