# EXP-0088 — M6-A27.8 Qwen3.8 L0 Q8_K contract

## Question

Does the L0 `ssm_out` projection discrepancy come from MIInfer's Q8_K
activation encoding, or from the subsequent Q5_K × Q8_K projection?

## Hypothesis

The external CPU reference and MIInfer may use different Q8_K rounding or
metadata rules even though both nominally use Q8_K activations.

## Baseline

EXP-0087 supplied the external P2 L0 gated output to MIInfer's production
`Q8_K → Q5_K` `ssm_out` path. The projected-output discrepancy remained:

```text
max_abs  0.00164509
RMS      2.38361e-05
```

## Method

The pinned llama.cpp reference is commit
`c0bc8591e8815c63cb01dd3f051a8b0df02501c9`. Its Q8_K reference quantizer is
`ggml/src/ggml-quants.c:quantize_row_q8_K_ref`; it uses 256-value blocks,
`-127 / max`, nearest-integer quantization, and 16 int16 block sums.

The existing MIInfer host Q8_K oracle uses the same contract. The diagnostic
replayed the external L0 P2 gated tensor through MIInfer, downloaded its full
7,008-byte Q8_K buffer, and compared it byte-for-byte with that reference
encoding. Production behavior was unchanged.

Command:

```bash
cmake --build build/mi50-release \
  --target miinfer-m6a21-qwen35-gpu-hybrid-block -j2
build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  /tmp/m6a273-reference --prefix64-l0-p2-output-projection
```

## Environment

Release build on the MI50/gfx906 development host. This was a correctness
diagnostic, not a performance benchmark.

## Results

| Encoding | Bytes | Fingerprint | Mismatched bytes |
| --- | ---: | ---: | ---: |
| MIInfer GPU replay | 7,008 | `9331021456029706823` | — |
| llama.cpp-compatible host reference | 7,008 | `9331021456029706823` | 0 |

The first-mismatch sentinel was `7008`, meaning no mismatch exists.

The external-gated replay still produced the same projected-output error as
EXP-0087:

```text
max_abs=0.00164509
mean_abs=4.84379e-06
RMS=2.38361e-05
relative_RMS=0.000114498
```

## Interpretation

Q8_K encoding is cleared for this contract. The remaining L0 discrepancy is
inside the Q5_K × Q8_K projection/accumulation path, or reflects a difference
between that GPU path and the pinned CPU dot-product ordering. It is not caused
by the activation quantizer or its metadata.

## Decision

**MEASUREMENT-ONLY / A27.8 complete.** No production behavior or tolerance
changed.

## Follow-up

Do not revisit Q8_K. If A27 needs another bounded diagnostic, compare the
Q5_K × Q8_K dot-product contract using the already identical Q8_K bytes and
external gated input. Otherwise close this representation branch and proceed
to the external observable-contract decision.
