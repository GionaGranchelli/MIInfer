# EXP-0087 — M6-A27.7 Qwen3.8 L0 output-projection contract adjudication

## Question

Does the L0 gated-output → `ssm_out` projection → residual path match the
external P2 contract, and where does its first discrepancy enter?

## Method

The existing L0 recurrent buffers were captured at P2. A shared-input replay
then supplied the authoritative external L0 gated output to the existing
MIInfer `ssm_out` projection path and residual add.

```bash
cmake --build build/mi50-release \
  --target miinfer-m6a21-qwen35-gpu-hybrid-block -j2
build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  /tmp/m6a273-reference --prefix64-l0-p2-output-projection
```

The external projected attention delta was derived as:

```text
external attn_residual - external layer input
```

## Contract clarification

L0 is a recurrent/DeltaNet layer. Its output projection is `ssm_out` using
the direct production path:

```text
gated FP32 [6144]
    ↓
Q8_K quantization [24 × 292 bytes]
    ↓
Q5_K × Q8_K GEMV
    ↓
projected FP32 [5120]
    ↓
residual add
```

The full-attention `o_input_f32_to_f16` and `o_output_f16_to_f32` stages do not
occur in this L0 path, so they were not fabricated for this test.

## Results

| Boundary / replay | Max abs | Mean abs | RMS | Relative RMS |
| --- | ---: | ---: | ---: | ---: |
| Production gated output vs external | 4.76837e-7 | 2.66646e-9 | 1.10165e-8 | 1.86333e-7 |
| Production projected output vs external | 0.00164509 | 4.84379e-6 | 2.38361e-5 | 0.000114498 |
| Production residual vs external | 0.00164509 | 4.84379e-6 | 2.38361e-5 | 0.000114070 |
| External gated input replay through MIInfer projection | 0.00164509 | 4.84379e-6 | 2.38361e-5 | 0.000114498 |
| External gated input replay through MIInfer residual | 0.00164509 | 4.84379e-6 | 2.38361e-5 | 0.000114070 |

The production Q8_K buffer contains 7008 bytes. Its diagnostic fingerprint is
`5689865191908132621`; the external-gated replay Q8_K buffer contains the same
7008-byte logical representation size and has fingerprint
`9331021456029706823`, as expected because the two replay inputs are distinct.

## Interpretation

The gated input is cleared: production gated output matches the external
checkpoint to roundoff. Supplying that same external gated output to MIInfer
does not remove the projection error, so the discrepancy is inside the L0
`ssm_out` output-projection contract—most likely the activation quantization or
the external reference's corresponding Q5_K activation representation.

The residual add is also cleared: its error is exactly inherited from the
projected output. This is the first concrete representation-contract mismatch
in the early P2 path and explains the initial L0 attention-residual error from
EXP-0086.

## Decision

**MEASUREMENT-ONLY / A27.7 complete.** The L0 output projection is not cleared
under the current external contract. No production behavior or tolerance
changed.

## Follow-up

Compare the external reference's exact `ssm_out` activation representation and
Q5_K matvec path against MIInfer's direct Q8_K path. Test one reference-
equivalent activation format/rounding boundary only; do not change the
residual or gated operation.
