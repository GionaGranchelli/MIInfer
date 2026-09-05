# EXP-0140 — M6-B48 persistent Q4_K FFN Down metadata

## Question

Can immutable Q4_K scale/minimum metadata be decoded once at model setup and
reused by the dominant FFN Down Q4_K×Q8_1 projections instead of being decoded
into LDS on every token?

## Baseline and candidate

The B41 production path stages Q8_1 input and decodes each pair of Q4_K
metadata values into LDS for every projection launch. B48 retained the raw
Q4_K payload, Q8_1 input, dot4 arithmetic, two-row mapping, and reduction, but
added an opt-in setup-time metadata decode and a persistent metadata buffer for
Q4_K FFN Down weights only.

The candidate was selected with
`MIINFER_Q4K_Q8_1_PERSISTENT_DOWN_METADATA=1`; the default remained B41.

## Environment

```text
GPU: AMD Instinct MI50 / gfx906
Clock: stable_peak
Model: /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Build: build/mi50-release
Fixture: /tmp/m6a273-reference-p12
```

## Baseline profile

The post-B47 production profile measured at position 63:

```text
total GPU event: 73.8226 ms/token
layer sum:       70.5030 ms/token
final LM head:    2.52352 ms
FFN Down:        approximately 0.42–0.45 ms/layer
allocations:     0
```

## Correctness

The candidate passed native 16-token generation and deterministic replay, the
64-layer external observable contract, poisoned reset/replay, and Release
CTest 20/20. Decode-loop allocations remained zero. Setup-time persistent
metadata increased tracked/peak device usage from `17,019,965,780` to
`17,242,788,180` bytes, an increase of `222,822,400` bytes.

## Results

Five-sample same-build medians:

| Workload | Control tok/s | Candidate tok/s | Change |
| --- | ---: | ---: | ---: |
| TG64 | 13.7638 | 13.7681 | +0.03% |
| TG128 | 13.5602 | 13.5376 | -0.17% |

The differences are below the useful threshold and do not show a repeatable
whole-token improvement. The extra persistent representation also costs about
223 MB of VRAM.

## Decision

**REJECT.** Persistent Q4_K metadata does not improve end-to-end decode on the
MI50 workload. The candidate was removed and B41 remains production-selected.

## Follow-up

Do not pursue further metadata-only variants without new profiling evidence.
The next candidate must target a different measured cost or eliminate actual
work rather than only relocating the same metadata.
