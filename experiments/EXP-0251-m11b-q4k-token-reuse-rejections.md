# EXP-0251 — M11-B native Q4_K token-reuse rejections

## Hypothesis

The proven four-wave/four-token Q4_K arithmetic can expose a larger logical
batch if the four output rows' weights are staged once in LDS, or if one
weight tile is held in registers while four-token microtiles accumulate into
partial output rows.

## Baseline

The exact `blk.8.ffn_down.weight` Q4_K shape `[17408,5120]`, using repeated
`launch_q4k_wave_gemv_batched4` launches. The B=64 control is sixteen launches.

## Candidates

1. A 256-thread block staged four complete native Q4_K rows (43.52 KiB LDS)
   and consumed B=4 token groups.
2. A 256-thread block walked the weight tiles first, held one tile in the
   existing Wave64 register mapping, and wrote partial output between tiles.

Both candidates retained the production Q4_K arithmetic and bounded the
logical batch to 4-token register microtiles.

## Environment

- GPU: AMD Instinct MI50, gfx906
- Model: `Qwen3.8-27B-Q4_K_M.gguf`
- Tensor: `blk.8.ffn_down.weight`, Q4_K `[17408,5120]`
- Build: `mi50-release`
- ROCm: 6.4.0 / LLVM 20
- Timing: interleaved HIP-event samples, 10 warmups and 51 rounds

## Correctness

Both candidates were finite and matched the repeated B=4 output. The LDS
candidate's maximum absolute error was `0`; the tile-major candidate's maximum
absolute error was `1.19209e-6`.

## Results

| Tokens | Repeated B=4 | LDS-staged | LDS / control | Tile-major | Tile / control |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 322.08 us | 781.12 us | 0.412x | 438.56 us | 0.725x |
| 8 | 626.40 us | 1119.36 us | 0.560x | 776.96 us | 0.791x |
| 16 | 1218.08 us | 1816.16 us | 0.671x | 1446.72 us | 0.837x |
| 32 | 2394.40 us | 3272.05 us | 0.732x | 2783.84 us | 0.859x |
| 64 | 4793.33 us | 6302.30 us | 0.761x | 5505.92 us | 0.871x |

## Interpretation

The native Wave64 row mapping does not benefit from serial token reuse. LDS
traffic/barriers dominate the first candidate; the tile-major partial-output
stores and serialized token groups still outweigh its reduced weight loads.

## Decision

**REJECT.** Both temporary kernels and the benchmark were removed. Further
native Q4_K row-tile variants are not justified. The next candidate must use a
different packed skinny-GEMM layout that exposes rows and tokens concurrently.
