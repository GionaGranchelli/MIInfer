# EXP-0293 — Async next-layer repacked-weight prefetch

## Hypothesis

Prefetching the next layer's packed MMQ weights on a non-blocking HIP stream
while the current layer computes will hide the measured P512 upload bucket
without retaining the whole model's repacked weights.

## Candidate

Each wide layer allocated a private six-buffer staging set, copied the next
layer's packed weights asynchronously, and handed them to the main stream via
HIP events.

## Environment

- AMD Instinct MI50 / gfx906
- Qwen3.8-27B-Q4_K_M
- exact P512 prompt, context capacity 1024
- full layer-major, B512 projections, row-128 MMQ
- recurrent FFN residency enabled

## Correctness and result

The one-token continuation remained valid (`brown`), but P512 fell to
61.09 tok/s (8,380.96 ms) from the prior 81.44 tok/s interleaved resident
result. Allocation stayed 29,956,706,644 B.

## Decision

**REJECT.** The second-stream/pageable-host transfer path adds contention or
blocking that outweighs any overlap. Remove it; retain synchronous upload
timing in the profiler and do not retry without pinned-host staging evidence.
