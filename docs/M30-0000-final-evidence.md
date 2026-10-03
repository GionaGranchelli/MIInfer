# M30-0000 — Integrated Evidence

## Required completion report

| Identifier | Value |
|---|---|
| `M30_0000_BASE_SHA` | `23deab2eda689fc0cfc21782d3f6df1303bee2ee` |
| `AGENT_WORKLOAD_VERSION` | `m30-agent-v1` |
| `BASELINE_FINAL_SHA` | `6dc66c59f7545a10041ad6984bc43ce4ad487740` |
| `REUSE_MAP_FINAL_SHA` | `97dd684b329617f6a8e8e58bfad70db96c050adb` |
| `INTEGRATED_EVIDENCE_SHA` | populated after this synthesis commit |

## Opportunity synthesis

| Mechanism | Repeated work today | Potential work avoided | State cost | Dependency | Evidence |
|---|---:|---:|---:|---|---|
| Exact-prefix KV reuse | 2,069,535 aggregate repeated-prefix tokens | 2,069,535 aggregate prefix tokens | 65,536 B/token for GQA | exact token/model/layout match; live owner | raw 86-event server log |
| GDN recurrent-state reuse | included in the same 2,069,535-token aggregate | GDN prefix scan/convolution within that aggregate; layer split not separately observable | 158,859,264 B fixed (151.50 MiB) | matrix, convolution history and position at the prefix | Workstream B source map |
| Append-only suffix execution | 3,651,780 logical prompt tokens versus 1,582,245 dispatched suffix tokens | 2,069,535 prefix tokens | GQA plus GDN checkpoint state | contiguous exact prefix and final-logit correctness | raw telemetry plus B map |
| Shared-prefix pages | no measured production path | not established | unknown | ownership/refcount/COW design absent | no current production evidence |
| Tail-Replay | no measured production path | not established | unknown | no current production evidence | no current production evidence |

The measured opportunity is aggregate model-token work; it must not be
presented as a 1:1 wall-clock prediction. The known zero-suffix/final-hidden
correctness fault remains a future enablement blocker.

## Reusable-state accounting

| Retained context | GQA KV | GDN state + history | Combined |
|---:|---:|---:|---:|
| 8K | 512.0 MiB | 151.50 MiB | 663.5 MiB |
| 32K | 2,048.0 MiB | 151.50 MiB | 2,199.5 MiB |
| 64K | 4,096.0 MiB | 151.50 MiB | 4,247.5 MiB |
| 128K | 8,192.0 MiB | 151.50 MiB | 8,343.5 MiB |

At the observed final context of 82,087 tokens, GQA KV accounting is
5,379,653,632 B; the fixed GDN state is 158,859,264 B. The selected MI50
reported 34,342,961,152 B total VRAM; the peak workload sample used
34,323,197,952 B.

## Exactly one next capability

The smallest next capability that unlocks the largest measured reduction is:

**exact-prefix KV reuse with suffix-only execution**, after correctness
hardening and qualification of the zero-suffix/final-hidden boundary.

Do not implement it in M30-0000. This is the sole recommendation for
M30-0001.

`M30_REUSE_OPPORTUNITY_MAPPED`
