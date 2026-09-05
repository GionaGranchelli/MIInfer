# EXP-0118 — M6-B26 Q-projection Q8_1 MMVQ candidate

## Question

Can full-attention Q projection use the existing Q4_K×Q8_1 MMVQ primitive
instead of the production Q4_K×Q8_K path?

## Baseline

Production B23 uses Q8_K activation quantization for full-attention Q
projection. The candidate was opt-in with:

```text
MIINFER_Q4K_Q8_1_MMVQ_ATTN_Q=1
```

The candidate changed only the Q projection input representation and reused
the existing Q4_K×Q8_1 MMVQ kernel. Production default remained unchanged.

## Environment

```text
GPU: AMD Instinct MI50 / gfx906
Clock policy: stable_peak (1725 MHz SCLK / 1000 MHz MCLK observed)
Model: /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Build: build/mi50-release
ROCm: 7.1.52802-9999
Fixture: /tmp/m6a273-reference-p12
```

## Correctness

* Candidate native 16-token replay: **FAIL**.
* Candidate external observable contract: **FAIL**.
* Candidate poisoned-reset replay: **FAIL**.
* At P0, final hidden relative RMS was approximately 0.771 and cosine 0.6435.
* At P0, logits cosine was 0.9465 and top-5 overlap was 2/5.
* The unchanged control rebuilt and passed native 16-token replay with zero
  decode allocations.

Because the candidate failed correctness immediately, no throughput A/B was
run and no performance claim is made.

## Interpretation

The existing Q4_K×Q8_1 MMVQ primitive is not a drop-in replacement for this
full-attention Q4_K×Q8_K Q projection under the accepted Qwen3.8 external
contract. The representation difference is too large to treat as a harmless
reduction-order variation. This is distinct from B23, which retained the
Q6_K×Q8_K contract and passed its observable checks.

## Decision

**REJECT.** The opt-in candidate was removed; production continues to use the
Q4_K×Q8_K Q projection. No new environment flag remains.

## Follow-up

Do not retry this representation-only Q-projection port. The next candidate
must preserve Q8_K semantics or be validated as a new external-contract
experiment with a concrete correctness rationale. The B25 fine attribution
identifies Q head normalization (~0.085 ms/layer) as the next separable
full-attention support cost to investigate.
