# EXP-0219 — M11-B Q4_K LDS Tile-Reuse Rejection

## Hypothesis

Staging four output rows of each Q4_K tile in LDS can reuse weights across
eight prompt vectors without eight live accumulators per wave. This is a
different decomposition from both EXP-0211's one-row LDS mapping and
EXP-0218's eight-accumulator mapping.

## Candidate

The candidate uses four output rows per block. The first variant uses 16
Wave64 waves, two token accumulators per wave, and 1,024 threads. The tighter
variant uses eight waves, four token accumulators per wave, and 512 threads.
Each tile's four row weights are copied once to LDS and consumed by all eight
token results.

## Environment

- GPU: AMD Instinct MI60/MI50-visible gfx906
- SCLK/MCLK: 1606/1000 MHz
- ROCm: 6.4.0 / LLVM 20
- Build: `mi50-release`
- Model: `Qwen3.8-27B-Q4_K_M.gguf`
- Shape: `blk.0.ffn_gate.weight`, 5,120 × 17,408 Q4_K
- Input: eight deterministic Q8_1 vectors
- Repetitions: 1 warmup, 6 batches, 2 iterations per batch

## Correctness

Both variants matched two B=4 launches with max absolute error `1.43051e-6`.

## Results

| Mapping | Median | Relative to two B=4 |
| --- | ---: | ---: |
| Two B=4 launches | 482.432–483.232 us | 1.00x |
| 1,024-thread LDS / 2 accumulators | 610.480 us | 0.79x |
| 512-thread LDS / 4 accumulators | 633.264 us | 0.76x |

## Decision

**REJECT.** Tile reuse halves the intended weight traffic, but LDS barriers,
the larger workgroup, and the extra staging path cost more than the saved
loads on the production shape. The runtime remains B=4.

## Follow-up

Further B=8 work needs a fundamentally different fused quantized GEMM or
persistent tile schedule; another LDS arrangement in this GEMV family is not
justified.
