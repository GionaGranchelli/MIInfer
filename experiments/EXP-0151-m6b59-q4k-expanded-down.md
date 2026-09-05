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
Down layers retain their existing path. It is production-selected by default;
set `MIINFER_Q4K_EXPANDED_DOWN=0` to use the packed control path.

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
| TG64 (initial) | 14.0802 | 14.1874 | +0.76% |
| TG128 (initial) | 13.8621 | 13.9449 | +0.60% |
| TG64 (fresh A/B) | 14.1161 | 14.1819 | +0.47% |
| TG128 (fresh A/B) | 13.8494 | 13.9308 | +0.59% |

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

**KEEP / production-selected.** The fresh A/B confirms a repeatable TG128 gain
and a positive TG64 result. The VRAM tradeoff is explicit and the packed path
remains available as a control.

The extra persistent storage is accepted for this 32 GB target; use the packed
control when a deployment cannot afford the additional ~3.08 GB.

## Follow-up

Revisit the selection only if context capacity or broader model coverage makes
the VRAM tradeoff unacceptable.
