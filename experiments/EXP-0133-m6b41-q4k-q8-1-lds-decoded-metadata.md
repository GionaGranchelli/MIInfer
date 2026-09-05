# EXP-0133 — M6-B41 Q4_K×Q8_1 decoded metadata staging

## Question

Can Q4_K scale/minimum metadata be decoded once per block into the existing
LDS tile, removing repeated packed-metadata decoding while preserving the
accepted Q4_K×Q8_1 arithmetic?

## Baseline and candidate

The baseline is B39: two independent 128-thread row reductions in a
256-thread workgroup sharing an LDS-resident Q8_1 activation tile and raw
Q4_K metadata. The candidate preserves the same weights, input, row mapping,
dot-product operations, and reduction order. One thread per block decodes the
eight scale/minimum pairs into a compact LDS metadata record; all lanes reuse
those values. The candidate was opt-in during validation and is now the
default; `MIINFER_Q4K_Q8_1_LDS_DECODED_METADATA=0` selects B39.

## Environment

```text
GPU:       AMD Instinct MI50 / gfx906
Clock:     stable_peak (1725 MHz SCLK / 1000 MHz MCLK observed)
Model:     /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Build:     build/mi50-release
Fixture:   /tmp/m6a273-reference-p12
Date:      2026-09-05
```

## Correctness

The candidate passed native 16-token generation and deterministic replay. The
complete P64 external observable contract passed without tolerance changes,
including poisoned reset/replay. Decode allocations remained `0`, and tracked
and peak device bytes remained `17,019,965,780`.

## Benchmark

Five-sample medians per process at stable_peak:

| Workload | B39 control median | Candidate median | Change |
| --- | ---: | ---: | ---: |
| TG64 ms | 4775.53 | 4601.58 | -3.64% |
| TG64 tok/s | 13.4017 | 13.9083 | +3.78% |
| TG128 ms | 9716.91 | 9354.72 | -3.73% |
| TG128 tok/s | 13.1729 | 13.6829 | +3.87% |

## Interpretation

Decoded metadata staging is a repeatable whole-token improvement at both
required decode lengths. It is materially different from B37's rejected full
weight staging: Q4 nibble loads remain global and only the small decoded
metadata record is shared.

## Decision

**KEEP; production-selected.** Enable decoded Q4_K metadata staging by
default. The environment flag above remains an A/B control.

## Follow-up

Refresh the post-B41 production profile and compare the updated result against
the pinned llama.cpp baseline. Do not repeat rejected full-weight staging or
unmotivated geometry variants.
