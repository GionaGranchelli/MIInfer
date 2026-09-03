# EXP-0074 — M6-A26.4 L30 production operand attribution

## Status

COMPLETE — production operand capture and one-at-a-time substitution. A26
remains RETEST; no tolerance or production execution change was made.

## Question

Which L30 recurrent operand differs from the external contract at P19→P20 and
causally explains the state discrepancy?

## Baseline and method

The common 32-layer GPU executor captured L30 operands immediately before the
P19 state update. A single unfused llama.cpp reference fixture supplied both
the P0 initialization/checkpoints and the P19/P20 `q_in`, `k_in`, `v_in`,
`b_in`, and `g_in` operands. The reference source was temporarily configured
to expose the non-fused recurrence operands, then restored.

The current MIInfer recurrence uses 16 key heads and 48 value heads. Operand
comparisons use the first 16 external q/k heads. The external `g_in` is
compared after the established `exp(g_in)` conversion as the production
`decay` operand.

## Environment and command

* model: `/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf`
* device: MI50 / gfx906
* reference: llama.cpp `c0bc8591e8815c63cb01dd3f051a8b0df02501c9`
* fixture: temporary `/tmp/m6a264-small.mmhOmI`

```bash
build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  /tmp/m6a264-small.mmhOmI /tmp/m6a264-small.mmhOmI \
  --prefix32-operand-attribution
```

## Production operand comparison

| Operand | Max abs | Mean abs | RMS | Relative RMS | Max index |
| --- | ---: | ---: | ---: | ---: | ---: |
| previous state | 0.0541533 | 8.87843e-05 | 2.23449e-04 | 3.39102e-02 | 86850 |
| q_in | 0.736532 | 4.55285e-02 | 8.05802e-02 | 1.03143e+01 | 1374 |
| k_in | 0.0180074 | 2.27402e-03 | 3.22484e-03 | 3.64849e-02 | 40 |
| v_in | 0.0162683 | 9.06639e-04 | 1.50051e-03 | 2.72016e-02 | 1208 |
| beta | 0.0118016 | 2.78970e-03 | 3.93496e-03 | 8.45948e-03 | 22 |
| g_in as decay | 0.00322658 | 4.41733e-04 | 6.97158e-04 | 7.40358e-04 | 18 |

## One-at-a-time GPU substitutions

Each row starts from the captured production operands and replaces only the
named operand with the external value before running the same MIInfer GPU
state-update primitive.

| Variant | State max abs | Mean abs | RMS | Relative RMS | Max index |
| --- | ---: | ---: | ---: | ---: | ---: |
| production | 0.0431299 | 8.95082e-05 | 2.25187e-04 | 3.36445e-02 | 86790 |
| external previous state | 0.0435030 | 3.00970e-05 | 1.57272e-04 | 2.34975e-02 | 86790 |
| external q_in | 0.0431299 | 8.95082e-05 | 2.25187e-04 | 3.36445e-02 | 86790 |
| external k_in | **0.00747788** | 8.46614e-05 | 1.75268e-04 | 2.61862e-02 | 666211 |
| external v_in | 0.0429355 | 8.50395e-05 | 2.18300e-04 | 3.26155e-02 | 86790 |
| external beta | 0.0431318 | 8.94087e-05 | 2.24989e-04 | 3.36148e-02 | 86790 |
| external g_in as decay | 0.0434681 | 8.95106e-05 | 2.25344e-04 | 3.36680e-02 | 86790 |

## Interpretation

The largest causal improvement comes from replacing `k_in`: state max error
falls from `0.0431299` to `0.00747788`. Replacing the previous state changes
the max error negligibly, and q substitution cannot affect the stored state in
this recurrence because q is consumed only by the recurrent-output path.
The remaining beta/value/decay substitutions are also largely neutral.

The result identifies the L30 key-input production path as the next upstream
trace target. It does not by itself prove that every `k_in` difference is a
semantic bug: representation/layout and the external checkpoint boundary
still need to be adjudicated.

## Decision

**KEEP / A26.4 complete.** The production operand attribution is complete and
the next evidence-backed target is the L30 input → normalization → K
projection → normalization boundary. No tolerance was changed and no 64-layer
composition was started.

## Follow-up

Trace only production L30 `k_in` upstream against the matching external
boundary. Do not reopen recurrence storage/update mechanics unless that trace
shows the stored operand is already correct.
