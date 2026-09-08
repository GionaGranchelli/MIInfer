# EXP-0266 — M16-C serving concurrency qualification

## Question

Can the bounded single-worker MI50 server provide predictable concurrent-client
behavior, and does evidence justify continuous batching?

## Harness repair

Earlier foreground output was not retained by the runner. A later captured run
silently lost its output; the first repaired attempt also exposed non-JSON-safe
control-probe bytes. `scripts/bench-serve-concurrency.py` now owns the server
with `subprocess.Popen`, writes separate logs/metrics and atomic JSON artifacts,
and records pre-readiness failures. The valid primary artifact is
`results/m16c/20260908-222755-d486f3e/`.

## Environment and workload

MIInfer `d486f3e`; Qwen3.8-27B Q4_K_M; MI50 gfx906, SCLK 1606 MHz, MCLK
1000 MHz, manual 225 W profile. The request was a deterministic ChatML user
prompt, streaming greedy generation, `max_tokens=64`, with three trials each.

## Results

| Clients | Accepted | HTTP 503 | Median latency (s) | Median TTFT (s) | Output hashes |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3 | 0 | 1.885 | 0.543 | 1 |
| 2 | 6 | 0 | 2.838 | 1.491 | 1 |
| 4 | 12 | 0 | 4.748 | 3.394 | 1 |
| 8 | 24 | 0 | 8.554 | 7.202 | 1 |
| 16 | 27 | 21 | 9.533 | 8.172 | 1 |

C16 accepted exactly one active plus eight pending requests per trial and
rejected the remaining seven. All C8/C16 health, ready, metrics, and models
probes returned HTTP 200 below one millisecond.

## Stream A/B

The retained C1/C4 non-stream artifact is `results/m16c/20260908-222137-d486f3e/`;
the corrected stream artifact is `results/m16c/20260908-223103-d486f3e/`.

| Clients | Stream median (s) | Non-stream median (s) |
| ---: | ---: | ---: |
| 1 | 1.892 | 1.895 |
| 4 | 4.741 | 4.747 |

## Decision

KEEP SERIALIZED. Completion hashes were identical across all accepted requests;
the queue contract and control plane remained stable. Aggregate throughput is
intentionally serialized and TTFT grows with queue position, but no scheduling
bubble or response-mode penalty justifies continuous batching for v0.2.0.

M16-C is PASS. Future batching requires a separate measured hypothesis.
