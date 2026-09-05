# EXP-0113 — M6-B20 Q6_K×Q8_1 MMVQ recurrent QKV

## Question

Can the existing Q6_K×Q8_1 MMVQ primitive replace the recurrent QKV
Q6_K×Q8_K projection for the `10240 × 5120` shape?

## Candidate

The opt-in candidate quantized the normalized recurrent input once to Q8_1
and used the existing 128-thread/output-row Q6_K×Q8_1 MMVQ launcher for QKV.
All other paths, including B19 Gate/Up and B18 FFN Down, were unchanged.
The control remained the production Q6_K×Q8_K QKV path.

## Environment

```text
GPU: AMD Instinct MI50 / gfx906
Clock policy: stable_peak (1725 MHz SCLK / 1000 MHz MCLK observed)
Model: /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Build: build/mi50-release
ROCm: 7.1.52802-9999
Compiler: Clang 20.0.0.rocm
Fixture: /tmp/m6a273-reference-p12
```

## Result

No performance measurement was accepted. The candidate failed the first
native 16-token replay with `native generation replay mismatch`. The external
observable run also showed immediate large discrepancies: for example,
position 0 produced a final hidden max error of approximately `96.1`, negative
logit cosine, and GPU argmax `58181` instead of reference token `11`.

The failure propagated across recurrent states from early positions, rather
than resembling the bounded low-margin differences accepted for the current
production path.

## Decision

**REJECT.** Remove the candidate from production and retain the existing
Q6_K×Q8_K recurrent QKV path. The Q6_K×Q8_1 MMVQ primitive is not a drop-in
replacement for this QKV tensor/layout contract. Do not retry without a new,
specific weight-layout or arithmetic-contract hypothesis.

## Checks

After removal, the target rebuilt successfully. B19's previously validated
default remains selected; no QKV candidate flag remains.
