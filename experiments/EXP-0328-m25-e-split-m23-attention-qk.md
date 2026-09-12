# EXP-0328 — M25-E split M23 attention Q/K geometry

## Hypothesis

The pinned gfx906 implementation's separate Q+gate and K launches, using
`BM=64` and `BN=128`, could beat MIInfer's fused M23 Q/K launch.

## Baseline

Current MIinfer M23 Q/K prefill at layer 3, using the fused
`13312 x 5120` Q/K projection and the row-128 dispatch.

## Candidate

An opt-in `MIINFER_PREFILL_SPLIT_ATTN_QK=1` path packed Q and K separately,
used the existing M23 Q8 activation layout, and wrote both results into the
same Q/K output slab with the pinned `64 x 128` launch geometry.

## Environment

- AMD Instinct MI50 / gfx906 / Wave64
- Qwen3.8-27B-Q4_K_M GGUF, SHA-256
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- Release build; ROCm `7.1.52802-9999`; clang `20.0.0.rocm`
- layer 3, B512, resident wide M23 attention path, Mx Q8 batch disabled

## Benchmark

Three interleaved control/candidate pairs ran as fresh processes. The
candidate was clock-stable enough for the layer bakeoff and completed without
an execution error.

## Correctness

Both paths were finite and had identical reported scalar parity:
`max_abs=0.033`, `rmse=0.001`.

## Results

| path | GPU ms | tok/s |
| --- | ---: | ---: |
| control | 63.3308, 63.3639, 63.3812 | median `8080.31` |
| split M23 | 68.2846, 68.3764, 68.3307 | median `7497.03` |

The split path was `7.84%` slower and used `55,377,920 B` more tracked layer
memory because the minimal candidate retained the fused Q/K allocation for
the existing decode contract.

## Interpretation

Matching the external grid alone is not a faithful performance port. The
external path also uses its own Q8 repacked activation and weight-layout
contract; reusing MIInfer's M23 tile and activation layout loses to the
existing row-128 fused path.

## Decision

**REJECT and remove.** Keep the fused M23 Q/K path selected.

## Follow-up

Do not revisit this geometry without porting the complete pinned data-layout
and execution contract.
