# EXP-0077 — M6-A26.7 L29 gated-output provenance

## Status

COMPLETE — L29 P19 post-recurrent normalization and gate boundaries traced.
A26 remains RETEST; no production behavior or tolerance changed.

## Question

Does the L29 gated-output discrepancy come from recurrent output
normalization, the gate projection, or their combination?

## Method

The common 32-layer GPU executor captured L29 at P19 at:

```text
recurrent output → head RMS normalization → ssm norm scaling
→ gate projection → SiLU(gate) → final gated output
```

The direct gate checkpoint was exported from the pinned llama.cpp reference
at commit `c0bc8591e8815c63cb01dd3f051a8b0df02501c9`. The external head-norm
and scaled values were reconstructed from the external recurrent output and
the model's resident `ssm_norm` weights, using the GPU kernel's pairwise
reduction order.

## Environment and command

* model: `/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf`
* device: MI50 / gfx906
* external fixture: temporary `/tmp/m6a267-reference-p19d`

```bash
build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  /tmp/m6a264-reference.YQO1ad /tmp/m6a267-reference-p19d \
  --prefix32-l29-gate-attribution
```

## Results — L29 P19

| Stage | Max abs | Mean abs | RMS | Relative RMS |
| --- | ---: | ---: | ---: | ---: |
| recurrent output | 0.000243431 | 0.00000857161 | 0.0000166008 | 0.0192208 |
| head norm | 0.155987 | 0.0065612 | 0.0117412 | 0.0238241 |
| head scaled | 0.180360 | 0.00681626 | 0.0124117 | 0.0279557 |
| gate projection | 0.105175 | 0.0232257 | 0.0291359 | 0.0189930 |
| SiLU(gate) | 0.0923302 | 0.0125703 | 0.0190340 | 0.0259747 |
| gated output | 0.0743616 | 0.00312962 | 0.00545215 | 0.0536301 |

## Interpretation

The stored recurrent output remains close, but normalization amplifies small
per-head differences, reaching `0.155987` at the normalized boundary. The
resident-scale boundary reaches `0.180360`, and the independently compared
gate projection also differs by `0.105175`; both effects contribute to the
final gated discrepancy. This does not identify a new state-storage or
state-update defect. It establishes that the remaining A26 issue is an
external-contract/input sensitivity question upstream of L29's final output,
not a GPU recurrent update failure.

## Decision

**MEASUREMENT-ONLY / A26.7 complete.** No production change was selected and
no tolerance was widened.

## Follow-up

Stop broad GPU state debugging. Adjudicate the external recurrent-state
contract or trace only a concrete upstream L29 input/gate differential before
resuming full composition.
