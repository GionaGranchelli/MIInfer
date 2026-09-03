# EXP-0075 — M6-A26.5 L30 K-path provenance

## Status

COMPLETE — L30 entry through K projection and K normalization traced. A26
remains RETEST; no tolerance or production behavior was changed.

## Question

Where does the L30 production `k_in` difference first appear on the path from
the L30 input through normalization, QKV projection, convolution/SiLU, and
K normalization?

## Method

The common 32-layer GPU executor captured L30 buffers at position 19:

```text
L30 input
  → attention RMSNorm
  → QKV projection
  → convolution + SiLU, K slice
  → K head L2 normalization
```

The external comparison used the same pinned llama.cpp fixture and commit as
EXP-0074. The external K slice was derived from `conv_output_raw-30` with the
reference SiLU operation; the external normalized K used the first 16 heads of
`k_in-30`, matching MIInfer's recurrent state-update contract.

## Environment and command

* model: `/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf`
* device: MI50 / gfx906
* reference: llama.cpp `c0bc8591e8815c63cb01dd3f051a8b0df02501c9`
* fixture: `/tmp/m6a264-reference.YQO1ad`

```bash
build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  /tmp/m6a264-reference.YQO1ad /tmp/m6a264-reference.YQO1ad \
  --prefix32-k-path-attribution
```

## Results — L30 P19

| Stage | Max abs | Mean abs | RMS | Relative RMS | Max index |
| --- | ---: | ---: | ---: | ---: | ---: |
| layer input (`l_out-29`) | 0.0793692 | 0.0157497 | 0.0198059 | 0.0179642 | 3938 |
| attention RMSNorm | 0.0892319 | 0.0140295 | 0.0178494 | 0.0211683 | 3941 |
| QKV projection | 0.168015 | 0.0281191 | 0.0354110 | 0.0350153 | 771 |
| K after convolution + SiLU | 0.0197068 | 0.00132717 | 0.00224709 | 0.0362777 | 40 |
| K after head L2 normalization | 0.0180074 | 0.00227402 | 0.00322484 | 0.0364849 | 40 |

The diagnostic also reports the external and GPU values at each row's maximum
error index.

## Interpretation

The first observed discrepancy is already present at the L30 input: current
MIInfer L29 output differs from external `l_out-29` by `0.0793692`. L30
normalization and QKV projection therefore cannot be treated as the source in
isolation. The QKV difference reaches `0.168015`, but the convolution/SiLU K
slice reduces the K-path difference to `0.0197068`, and K normalization changes
it only slightly to `0.0180074`.

This does not identify a standalone L30 K-kernel defect. It establishes that
the next upstream boundary is L29 output provenance. The L30 K path is not
where the first GPU/reference separation occurs.

## Decision

**MEASUREMENT-ONLY / A26.5 complete.** No production change was selected and
no tolerance was widened. A26 remains RETEST.

## Follow-up

Trace L29 output provenance, beginning with the L29 input and its final
residual/FFN path. Reuse the same boundary method and stop if the first
divergence is localized. Do not reopen L30 state storage or update mechanics.
