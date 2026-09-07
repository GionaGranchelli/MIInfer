# EXP-0213 — M11-B FP16 GEMM with On-Device K-Quant Dequantization

## Hypothesis

Dequantizing a native Q4_K/Q6_K FFN Down matrix to FP16 and using a batched
hipBLAS GEMM could replace repeated B=4 GEMV launches during layer-major
prefill and reach the M11-B P512 target.

## Motivation

EXP-0211 and EXP-0212 rejected the tested B=8 and row-LDS GEMV mappings. A
real skinny-GEMM upper-bound test was required before adding more runtime
batching machinery.

## Baseline

Qualified layer-major B=4 prefill on the Qwen3.8-27B-Q4_K_M model:

- P512 median: 12,812.97 ms / 39.96 tok/s
- P128: 3,106.36 ms / 41.21 tok/s

## Candidate

For each layer-major chunk, convert the FFN Down activation to FP16, expand
the native packed weight to an FP16 matrix, then run `hipblasGemmEx` with FP32
accumulation. The candidate was wired only behind
`MIINFER_PREFILL_FP16_GEMM=1` and was removed after measurement.

## Environment

- GPU: gfx906 MI50/MI60-visible device
- SCLK/MCLK: 1606/1000 MHz
- ROCm: 6.4.0 / LLVM 20
- Model: `/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf`
- Dominant tensor: Q4_K `blk.8.ffn_down.weight`, 5120 × 17408
- Candidate mode: `MIINFER_PREFILL_LAYER_MAJOR=1`, `MIINFER_PREFILL_FP16_GEMM=1`, `MIINFER_HIP_GRAPH=0`

## Correctness

The isolated Q4 expansion plus FP16 GEMM matched the host-dequantized FP16
reference for the sampled row with max absolute error `2.83e-7`. The native
Q8 GEMV comparison differed by `0.0131` max absolute error, as expected from
the distinct activation quantization path. A P1 runtime check still selected
`hello`; no production correctness qualification was claimed for the
aborted long run.

## Results

Single Q4_K projection, measured at B=128:

| Stage | Time |
| --- | ---: |
| Host dequantization | 3049.3 ms |
| GPU dequantization | 639.4 ms |
| FP16 GEMM | 3.66 ms |

The conversion dominates the GEMM by about 175×. The first opt-in runtime
check at P1 took 17,647.32 ms, and the P129 check remained active for more
than one minute before being stopped; it was not a production improvement.

## Interpretation

The FP16 GEMM itself is fast enough to be interesting, but repeatedly
materializing a 178 MiB matrix is not. A dequantize-then-GEMM runtime path
would need persistent weight caching or a fused quantized skinny-GEMM kernel;
the temporary workspace implementation does neither.

## Decision

**REJECT.** Remove the temporary runtime integration, API additions, and
benchmark. Keep the validated B=4 layer-major path as production candidate.

## Follow-up

Only a fused Q4_K/Q6_K batched GEMM that consumes the native layout directly,
or a measured memory-safe persistent weight cache, justifies another M11-B
implementation attempt.
