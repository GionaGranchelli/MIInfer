# EXP-0131 — M6-B39 Q4_K×Q8_1 metadata staging

## Question

Can the accepted Q4_K×Q8_1 LDS-input path avoid repeated per-lane reads and
decoding of Q4_K metadata by staging only `d`, `dmin`, and the 12 packed
scale/min bytes in LDS?

## Hypothesis

B37 showed that staging complete Q4_K blocks is too expensive. The metadata
is only 16 bytes per block, while Q4 nibble loads can remain global. Staging
metadata should reduce duplicate metadata traffic and unpack work without the
weight-traffic cost of B37.

## Baseline and candidate

The baseline is B35/B38: two independent 128-thread row reductions in a
256-thread workgroup sharing an LDS-resident Q8_1 activation tile. The
candidate keeps that geometry and dot/reduction order, but also stages the two
rows' Q4_K metadata in LDS. It was selected with
`MIINFER_Q4K_Q8_1_LDS_METADATA=1` during validation; the production default is
now enabled and `=0` selects the B38 control.

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

The candidate passed native 16-, 64-, and 128-token generation with exact
deterministic replay. The complete P64 external observable contract passed
without tolerance changes, including poisoned reset/replay. Decode
allocations remained `0`, and tracked/peak device bytes remained
`17,019,965,780`.

## Benchmark

Five-sample medians per process at stable_peak:

| Workload | B38 control median | Candidate median | Change |
| --- | ---: | ---: | ---: |
| TG64 ms | 5145.80 | 4775.53 | -7.19% |
| TG64 tok/s | 12.4373 | 13.4017 | +7.75% |
| TG128 ms | 10445.0 | 9716.91 | -6.97% |
| TG128 tok/s | 12.2547 | 13.1729 | +7.49% |

The candidate retains the same logical projection structure and adds only a
small metadata tile to the existing activation tile. It does not add weight
copies or persistent allocations.

## Interpretation

Metadata-only LDS staging is a repeatable whole-token improvement at both
generation lengths. The result is distinct from and substantially better than
B37's full-weight staging, which regressed about 3.9%.

## Decision

**KEEP; production-selected.** Q4_K×Q8_1 metadata staging is now enabled by
default. Set `MIINFER_Q4K_Q8_1_LDS_METADATA=0` to run the B38 control.

## Follow-up

Refresh the complete production profile against the pinned llama.cpp baseline.
Do not repeat Q4_K weight-staging or geometry variants without new evidence.
