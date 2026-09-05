# EXP-0123 — M6-B31 recurrent FFN Gate/Up two-row MMVQ

## Question

Can two independent recurrent FFN output rows share one 128-thread workgroup
in the Q4_K×Q8_1 MMVQ path without losing the accepted external contract?

## Baseline and candidate

Baseline: B30 production path at commit `fd77fa5`, with one 128-thread
workgroup reducing one output row. Candidate: two independent Wave64 rows per
128-thread workgroup, tested only for recurrent FFN Gate/Up. Quantization,
weights, Q8_1 blocks, and output precision were unchanged.

## Environment

```text
GPU:       AMD Instinct MI50 / gfx906
Clock:     stable_peak (1725 MHz SCLK / 1000 MHz MCLK observed)
Model:     /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Build:     build/mi50-release
Fixture:   /tmp/m6a273-reference-p12
```

## Results

The candidate preserved native replay and zero decode allocations, but the
external observable contract exposed a new P6 teacher-forced decision change:
`1746 → 3779`. The targeted P6 observable checkpoint had final-logit cosine
`0.999916`, but only `4/5` top-5 overlap; the reference winner was rank `2`
on the candidate. This does not satisfy the established A27 contract, which
requires the accepted low-margin flips to preserve the measured top-5 set.

The candidate was faster in the sampled end-to-end runs:

| Workload | Control tok/s | Candidate tok/s | Change |
| --- | ---: | ---: | ---: |
| TG64 | 11.8759 | 12.0485 | +1.45% |
| TG128 | 11.7205 | 11.8955 | +1.49% |

Both benchmark paths reported replay `PASS`, zero decode allocations, and
unchanged `17,019,965,780` tracked/peak device bytes. Native candidate
`--generate16` replay passed, but its last token changed to `6437`, consistent
with the P6 branch.

## Decision

**REJECT for production.** The candidate's speedup is real but correctness is
not within the pinned external observable contract. The two-row kernel and
its opt-in path were removed; B30 remains the production default.

## Follow-up

Do not retry this geometry without a new numerical-contract mechanism. Return
to the post-B30 profile and select a candidate that preserves the accepted
observable behavior.
