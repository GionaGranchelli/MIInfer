# EXP-0403 — M31 Q8-derived FP16 tiled attention

**Status:** IN PROGRESS

**Baseline:** `88fd63eb0cf8b765f37e5f52168c80783df62e7b`

## Objective

Compare the current direct Q8/Q8 Split-K attention with the existing
query-tiled FP16 KQ-fragment implementation at the final 512-token tile of a
64K context (`P=65024, S=512`). The FP16 tensors are dequantized once from the
same Q8 values and scales so this isolates attention execution and arithmetic.
Both routes use 32 KV splits.

## Gates

- Verify candidate BDF and compare outputs (`max_abs <= 0.01`, cosine
  `>= 0.9999`).
- Five interleaved timings on the same MI50 and operands.
- Treat this only as an attention-path result; cache-conversion and full-model
  memory costs are excluded. No runtime selector without a measured full-model
  follow-up and acceptable token behavior.

## Result

Pending.
