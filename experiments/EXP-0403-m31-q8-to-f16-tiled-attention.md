# EXP-0403 — M31 Q8-derived FP16 tiled attention

**Status:** ATTENTION GATE PASSED — FULL MODEL PENDING

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

On Machinist GPU2 (`0000:88:00.0`, gfx906), the Q8-derived FP16 route passed
the output gate: maximum absolute difference `0.0000573`, cosine `0.9999782`.
Five interleaved GPU-event samples were:

- Direct Q8/Q8: `1650.706, 1604.902, 1632.943, 1587.101, 1586.676 ms`; median
  `1604.902 ms`.
- FP16 tiled, 32 splits: `191.513, 191.811, 193.871, 191.941, 192.119 ms`;
  median `191.941 ms`, an 8.36x speedup.

Independent GPU2 junction polling peaked at 53 C. This is a synthetic attention
microbenchmark; it excludes model KV production, cache conversion/storage cost,
whole-model execution, logits and token behavior. The next gate is a matched
full-model prefill comparison using the built-in FP16 KV mode and the existing
opt-in tiled route. No production selector is enabled by this result alone.
