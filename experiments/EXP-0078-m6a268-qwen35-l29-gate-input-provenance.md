# EXP-0078 — M6-A26.8 L29 gate-input provenance

## Status

COMPLETE — the L29 gate projection was replayed with production and external
normalized inputs. A26 remains RETEST; no production behavior or tolerance
changed.

## Question

Is the L29 gate-projection discrepancy caused by the Q4_K×Q8_K projection
primitive or by the normalized activation entering it?

## Method

At L29/P19, the existing production gate projection was replayed twice:

```text
production normalized input → MIInfer gate projection
external normalized input   → MIInfer gate projection
```

Both outputs use the same resident L29 gate weights and the same production
Q4_K×Q8_K implementation. The results were compared with the external
`z-29` checkpoint from llama.cpp commit
`c0bc8591e8815c63cb01dd3f051a8b0df02501c9`.

## Environment and command

* model: `/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf`
* device: MI50 / gfx906
* production fixture: `/tmp/m6a264-reference.YQO1ad`
* external P19 fixture: `/tmp/m6a267-reference-p19d`

```bash
build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  /tmp/m6a264-reference.YQO1ad /tmp/m6a267-reference-p19d \
  --prefix32-l29-gate-attribution
```

## Results

| Gate input | Max abs | Mean abs | RMS | Relative RMS |
| --- | ---: | ---: | ---: | ---: |
| production normalized input | 0.105175 | 0.0232257 | 0.0291359 | 0.0189930 |
| external normalized input | **1.90735e-6** | **2.06199e-7** | **2.70959e-7** | **1.76632e-7** |

For context, one-at-a-time gated-output substitution gave:

| Substitution | Gated-output max abs |
| --- | ---: |
| production recurrent output + production gate | 0.0743616 |
| external recurrent output + production gate | 0.0735313 |
| production recurrent output + external gate | 0.0573288 |
| external recurrent output + external gate | 8.9407e-8 |

## Interpretation

The existing L29 gate projection is numerically compatible with the external
contract when supplied the external normalized input. The large gate
projection discrepancy is therefore inherited from the L29 normalized
activation, which is already different because the L29 input differs from the
external `l_out-28` checkpoint. The recurrent output remains close but its
head normalization is sensitive to that small difference. This is not a new
Q4_K×Q8_K kernel defect and not a recurrent state storage/update defect.

## Decision

**MEASUREMENT-ONLY / A26.8 complete.** No production change was selected and
no tolerance was widened.

## Follow-up

Stop broad GPU state-mechanics debugging. Make the explicit external state
contract decision using the bounded output/state error distribution, then
resume full GPU composition or trace only a newly demonstrated upstream
contract mismatch.
