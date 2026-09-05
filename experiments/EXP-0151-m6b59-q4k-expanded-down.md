# EXP-0151 — M6-B59 expanded Q4_K FFN Down weights

## Question

Can a model-load expansion of Q4_K FFN Down weights improve decode throughput
by replacing packed-nibble extraction with byte-addressable nibbles while
preserving the external Qwen3.8-27B execution contract?

## Candidate

The opt-in path expands each Q4_K block from 144 bytes to a 276-byte device
block containing decoded scale/minimum metadata and 256 byte-addressable
nibbles. The existing Q4_K×Q8_1 dot-product arithmetic and 256-thread
two-row reduction are retained. Only Q4_K FFN Down projections use it; Q6_K
Down layers retain their existing path. Production default remains unchanged.

Enable with `MIINFER_Q4K_EXPANDED_DOWN=1`.

## Environment

```text
GPU:       AMD Instinct MI50 / gfx906
Model:     /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Build:     build/mi50-release
Clock:     stable_peak; 1725 MHz SCLK / 1000 MHz MCLK
Fixture:   /tmp/m6a273-reference-p12
```

## Correctness

The first implementation exposed an incorrect expanded-nibble index and was
fixed before classification. The corrected candidate matched the control's
16-token and 64-token generation endpoints and state fingerprints. Both
paths passed deterministic replay with zero decode allocations. The external
64-layer observable-contract run completed with final observable PASS and
matching position-64 argmax; state-envelope diagnostics remain diagnostic,
as in the established contract.

Release build and the existing GPU generation checks passed. The full release
CTest suite remains the required final regression gate before production
selection.

## Results

Fresh five-sample medians:

| Workload | Control tok/s | Candidate tok/s | Change |
| --- | ---: | ---: | ---: |
| TG64 | 14.0802 | 14.1874 | +0.76% |
| TG128 | 13.8621 | 13.9449 | +0.60% |

| Resource | Control | Candidate | Delta |
| --- | ---: | ---: | ---: |
| Device bytes after setup | 17,019,965,780 | 20,094,914,900 | +3,074,949,120 |
| Decode allocations | 0 | 0 | 0 |

The candidate's extra persistent storage is about 3.08 GB, while total
device usage remains about 20.1 GB on the 32 GB MI50.

## Interpretation

Expanded Q4_K storage improves the measured decode path modestly and
consistently across TG64 and TG128. The gain is small but above the useful
threshold for a low-risk, model-load-only specialization. It is not a
universal kernel improvement: it trades approximately 3.08 GB of VRAM for
about 0.6–0.8% throughput.

## Decision

**RETEST / candidate promising; production selection pending final regression
and interleaved A/B confirmation.**

The production default remains the packed Q4_K path until those gates are
complete. Do not claim a final M6 performance result from this experiment.

## Follow-up

Run the full Release CTest suite and an interleaved TG64/TG128 A/B. If both
remain exact and the gain persists, production-select the expanded Down path
only for configurations with sufficient VRAM headroom; otherwise reject it.
