# EXP-0280 — M22 M13 B8 eight-wave mapping

## Hypothesis

The M13 B8 quantized kernel is register-limited by a four-wave block; using
eight Wave64s per block may recover its expected batch reuse.

## Configuration

Lab-only change to the M13 dispatch: B8 uses 512 threads and eight rows per
block. The production B4 dispatch was restored unchanged after measurement.
Raw result: `results/m22-m13-quant-mm/20260909-b512.json`.

## Result

| shape | B64 baseline/candidate us | candidate speed vs B4 | max error |
|---|---:|---:|---:|
| Q4_K FFN-down | 5,337.30 / 6,150.58 | 0.868x | 0.000000 |
| Q6_K QKV | 2,314.57 / 3,555.85 | 0.651x | 0.000000 |

Across B64/B128/B256/B512/B2048, Q4_K was 0.868/0.868/0.867/0.864/0.864x;
Q6_K was 0.651/0.651/0.667/0.669/0.669x. The wider block did not recover
the missing matrix reuse.

## Decision

REJECT. The experiment is retained as a negative result; no runtime kernel
change remains from it.
