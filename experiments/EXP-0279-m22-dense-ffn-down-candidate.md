# EXP-0279 — M22 dense FFN-down candidate

## Hypothesis

The existing optional FP16/rocBLAS dense FFN-down path will reduce the
dominant deferred tail enough to improve M12 prompt processing.

## Configuration

MIInfer release build with `MIINFER_PREFILL_LAYER_MAJOR=1` and
`MIINFER_PREFILL_DENSE_FFN_DOWN=1`, exact Qwen3.8-27B-Q4_K_M model, P512,
one-token PP-only measurement. Raw output:
`results/m22-candidates/20260909-204151-1514411/`.

## Result

| path | PP tok/s | prefill ms | peak VRAM |
|---|---:|---:|---:|
| M12 B4/B64 baseline | 45.92 | 11,148.96 | 24,977,998,164 |
| dense FFN-down | 46.62 | 10,982.36 | 30,052,385,108 |

The candidate improves P512 PP by only 1.5% while adding approximately 5.1
GB peak VRAM. It does not approach the mx reference's 222.55 tok/s.

## Correctness

The run completed successfully. Existing exact greedy token checks remain the
correctness gate; no divergence was observed in this one-token run.

## Decision

REJECT as the M22 production candidate. The small speed improvement is not a
reasonable memory trade and does not address the broader B4 quantized-tail
cost.
