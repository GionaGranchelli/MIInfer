# EXP-0089 — M6-A27.9 Qwen3.8 L0 Q5_K block contract

## Question

Is the L0/P2 `Q5_K × Q8_K` discrepancy only final reduction ordering, or do
individual 256-value block contributions differ from the scalar reference?

## Baseline

EXP-0088 cleared Q8_K encoding byte-for-byte. With the external L0/P2 gated
input and identical 7,008-byte Q8_K data, the existing MIInfer projection
still differed by `0.00164509` at the selected output row.

## Method

The diagnostic selected output row `3994`, the row with the largest baseline
projection error, and compared all 24 per-block contributions against the
scalar llama.cpp-compatible Q5_K × Q8_K formula. The scalar formula uses the
Q5 high-bit/scale/minimum unpacking, integer per-lane accumulation, Q8 block
sums, and FP32 scale application used by the pinned llama.cpp generic path.

The GPU diagnostic invoked the existing Q5_K × Q8_K kernel once per block so
that each result represented one production-device block contribution.

## Results before fix

Individual blocks disagreed, including:

```text
block 5   abs error 0.000894308
block 12  abs error 0.000897646
block 2   abs error 0.000231147
```

The block-sum error was `0.00164604`, so the discrepancy was not only the
final cross-block reduction.

## Change

Changed the gfx906 Q5_K × Q8_K kernel to accumulate quantized products in
int32 partials, apply Q5 scale/minimum terms and Q8 block sums using the
reference contract, and perform the final FP32 scale combination. The public
kernel interface and Q5/Q8 layouts are unchanged.

## Results after fix

| Check | Result |
| --- | ---: |
| Q8_K byte mismatches | 0 / 7,008 |
| Maximum per-block contribution error | `5.96e-8` |
| Block-sum error | `0` |
| External-gated projected output max error | `9.53674e-7` |
| External-gated residual max error | `9.53674e-7` |
| Production projected output max error | `1.78814e-7` |
| Release CTest | 20/20 |

## Decision

**KEEP / production change accepted.** The Q5_K × Q8_K path had a real
per-block arithmetic-contract mismatch, not merely final reduction ordering.
The integer accumulation change restores the pinned reference behavior at the
tested L0/P2 contract boundary.

## Follow-up

Rerun the existing 64-layer external observable-contract adjudication and
trajectory checks before closing A27. No further Q8_K investigation is
needed.
