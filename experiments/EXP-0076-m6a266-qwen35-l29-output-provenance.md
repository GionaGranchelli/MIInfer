# EXP-0076 — M6-A26.6 L29 output provenance

## Status

COMPLETE — L29 output provenance traced through the recurrent output, gated
attention path, residual, post-attention norm, FFN, and final residual. A26
remains RETEST; no production behavior or tolerance changed.

## Question

Where does the L29 output discrepancy already visible at the L30 input first
appear inside L29?

## Method

The common 32-layer GPU executor captured L29 at P19 at these boundaries:

```text
L29 input → attention norm → QKV → recurrent output → gated attention
→ attention residual → post-attention norm → FFN output → L29 output
```

The external values came from the pinned llama.cpp fixture at commit
`c0bc8591e8815c63cb01dd3f051a8b0df02501c9`.

## Environment and command

* model: `/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf`
* device: MI50 / gfx906
* fixture: `/tmp/m6a264-reference.YQO1ad`

```bash
build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  /tmp/m6a264-reference.YQO1ad /tmp/m6a264-reference.YQO1ad \
  --prefix32-l29-path-attribution
```

## Results — L29 P19

| Stage | Max abs | Mean abs | RMS | Relative RMS |
| --- | ---: | ---: | ---: | ---: |
| layer input | 0.225975 | 0.015012 | 0.0190463 | 0.0173931 |
| attention norm | 0.0784293 | 0.0138195 | 0.0174501 | 0.0209835 |
| QKV projection | 0.164217 | 0.0264573 | 0.0333143 | 0.0342477 |
| recurrent output | 0.000243431 | 0.00000857161 | 0.0000166008 | 0.0192208 |
| gated attention | 0.0743616 | 0.00312962 | 0.00545215 | 0.0536301 |
| attention residual | 0.0865796 | 0.0156088 | 0.0195976 | 0.0180454 |
| post-attention norm | 0.0802808 | 0.0140252 | 0.0177719 | 0.0302579 |
| FFN output | 0.0495043 | 0.0081437 | 0.0104944 | 0.0600800 |
| L29 output | 0.0793692 | 0.0157497 | 0.0198059 | 0.0179642 |

## Interpretation

The recurrent state output is extremely close to the external checkpoint.
The first large L29 discrepancy appears after recurrent-output gating, then
propagates through the attention residual and the rest of the layer. This
clears the L29 state update as the source of the L30 input mismatch and makes
the L29 gated-output path the next narrow diagnostic target.

## Decision

**MEASUREMENT-ONLY / A26.6 complete.** No production change was selected and
no tolerance was widened.

## Follow-up

Separate L29 head normalization from gate projection/SiLU provenance. Do not
reopen recurrent state storage or the update kernel.
