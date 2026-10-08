# M31-0006 — Production graph-decode baseline

Status: bounded direct-attention baseline measured at all requested shapes; short
real-model HIP-Graph decode smoke passed. These data are single-call exploratory
measurements, not stable release benchmarks.

The microbenchmark invokes the production
`launch_qwen35_tiled_online_attention_quant_dynamic` entry point, captures its
two HIP launches, and uses deterministic FP16 KV buffers (24 query heads, 4 KV
heads, D=256). It checks the eager/default path and captured graph against an
FP64 CPU oracle and poisons the first invalid KV position. Graph replay time is
event time; submit-to-completion also includes host launch/synchronization.

## Default schedule

| Context | Active splits | Kernel pair (ms) | Graph replay (ms) | Graph submit+sync (ms) | Static scratch | GPU allocation delta |
|---:|---:|---:|---:|---:|---:|---:|
| 8K | 16 | 0.717439 | 0.732319 | 0.753170 | 1,585,152 B | 192,937,984 B |
| 32K | 64 | 1.038398 | 1.039199 | 1.058092 | 1,585,152 B | 293,601,280 B |
| 64K | 64 | 1.989757 | 1.985757 | 1.994080 | 1,585,152 B | 427,819,008 B |
| 128K | 64 | 3.381915 | 3.297275 | 3.305768 | 1,585,152 B | 696,254,464 B |

Every default sample passed CPU-reference, graph/eager, and poison checks at
absolute tolerance `2e-3`; measured maximum FP64-reference error was below
`8e-10`. `MIINFER_ATTENTION_SPLITS` was unset for these controls. 128K here is
one direct attention shape only—no 128K model prefill was run.

The allocation delta is not scratch-only: it includes K/V plus HIP/runtime and
kernel allocations. Across all four contexts it equals K+V bytes plus a stable
159,379,456 B overhead. The source-accounted static split scratch is
1,585,152 B; attribution of the remaining constant overhead is still open.

## Real-model smoke

`miinfer-m31-0006-production-graph-decode-smoke` ran the Qwen3.8-27B-Q4_K_M
model on the Z840 MI50 with an 8-token prompt and four generated tokens. It
reported `used_hip_graph=true`, `prefill_ms=505.641`, `decode_ms=94.9625` for
three decode forwards, or 31.591 tok/s / 31.654 ms per decode forward. This is
a short-prompt smoke and not comparable to the long-context frontier. It proves
the graph-enabled production call completes; it does not isolate each
attention layer's share of end-to-end latency.

## Raw evidence

- [8K default](../../results/m31-0006-attention-baseline-8k-call1.log) — SHA-256 `ca0acdfceaf38a0316a4c442db4b32819c3c039895cb2b07d0f7ae3eff655e4c`
- [32K default and split-32](../../results/m31-0006-attention-baseline-32k-schedule-call1.log) — SHA-256 `db646e7587af0fd7d3207266c6d618a51cf3799d5cc5d7401edf50a51962754c`
- [64K default](../../results/m31-0006-attention-baseline-64k-call1.log) — SHA-256 `010bb3a00c2f3fab1a0e6e4839fd27aa138b840f72b1fad3ca2037eb191708d3`
- [128K direct shape](../../results/m31-0006-attention-baseline-128k-call1.log) — SHA-256 `c3f8f6dd438367ced160d0c181fc6aada30a675f6d69b232192ee1fbab802a96`
- [Production graph smoke](../../results/m31-0006-production-graph-decode-smoke.log) — SHA-256 `17b4ce26c88332dfe8cd247be90e2f9963e0c8212aacf02f0779a93bae4b1690`

Host, device, binary/build, temperature samples, stop-guard events, and cleanup
are retained in the raw logs. One sample per context is sufficient to screen
large differences; repeat the winner only if a candidate beats the default.
