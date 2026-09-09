# EXP-0282 — M22 SSM-output dense batch candidate

## Hypothesis

The recurrent Q5_K SSM-output projection is part of the deferred-tail cost;
expanding it to FP16 and using batched rocBLAS GEMM should improve the accepted
FFN-dense prefill path.

## Baseline and candidate

Both runs used the release MIInfer binary, exact Qwen3.8-27B-Q4_K_M model,
`MIINFER_PREFILL_LAYER_MAJOR=1`, and `MIINFER_PREFILL_DENSE_PROJECTIONS=1`.
The candidate additionally set `MIINFER_PREFILL_DENSE_SSM=1`. Raw artifacts:
`results/m22-candidates/20260909-213035-1916447/` and
`results/m22-candidates/20260909-213251-1958128/`.

## Results

| case | FFN candidate | FFN + SSM candidate |
|---|---:|---:|
| P512 PP tok/s | 55.78 | 57.31 |
| P2048 PP tok/s | 52.69 | 52.59 |
| P512 peak VRAM bytes | 29,973,676,372 | 31,011,766,612 |

The apparent P512 gain is not reproduced at P2048 and does not meet the
material-family gate. It also leaves little 32 GB VRAM headroom.

The same eight-token deterministic replay produced identical token IDs to the
FFN baseline:

```text
561 3841 13477 37550 14330 42903 4906 13
```

## Decision

REJECT the SSM dense path. Keep its implementation out of the runtime; retain
the raw measurements as a negative result. The accepted experimental path is
FFN gate/up densification only.

## Follow-up

Any further SSM work needs a same-size native representation or a measured
kernel-level win large enough to justify its memory cost.
