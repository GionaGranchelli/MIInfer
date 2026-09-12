# EXP-0337 — M25 P512 prefill HIP graph screen

**Status:** REJECT
**Milestone:** M25
**Date:** 2026-09-12

## Hypothesis

Capturing the resident Mx full-layer-major P512 sequence as one HIP graph
could recover the roughly 34 ms difference between the fresh diagnostic's
ordered layer timing and wall time.

## Candidate

The working-tree candidate captured the 64-layer resident Mx P512 body after
model allocation and before generation timing, then replayed it after the
existing 512 embedding launches. It was opt-in and restricted to the exact
resident Mx P512 vector. Kernels, weights, state representation, decode, and
shorter prompts were unchanged.

## Environment

- AMD Instinct MI50 / gfx906 / Wave64
- Qwen3.8-27B-Q4_K_M, model SHA-256
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- Release `build/mi50-release`
- explicit M25-H/I vector, `MIINFER_MX_Q8_BATCH=0`
- clean `env -i` processes, exact repeated-fox P512 prompt
- benchmark: `--max-tokens 0 --no-stream`
- no claim of clock qualification; this was a screen only

## Correctness

The graph candidate passed the same-process gate:

```text
first_token=13477 (brown)
continuation_token=37550
first_prefill_ms=2431.17
repeat_prefill_ms=2534.39
```

The candidate and control reported the same MIInfer allocation counter,
`21,993,242,964 B`. `hipMemGetInfo` showed the graph process with about
`8,388,608 B` less free VRAM, which is not included in the internal counter.

## Results

Three fresh process pairs, interleaved across the screen, produced:

| path | P512 samples (ms) | median ms | median tok/s |
| --- | --- | ---: | ---: |
| control | 2425.11, 2382.17, 2471.90 | 2425.11 | 211.12 |
| prefill HIP graph | 2416.16, 2451.35, 2435.88 | 2435.88 | 210.21 |

The candidate was `10.77 ms` slower by median (`0.44%`) and consumed an
additional approximately `8 MiB` of reported free-VRAM budget.

## Decision

**REJECT.** The graph does not recover a stable P512 win on the current MI50
workload. Remove the experiment and retain host-submitted full-layer-major
prefill. Do not revisit this graph shape without a new measured scheduling
or graph-memory hypothesis.
