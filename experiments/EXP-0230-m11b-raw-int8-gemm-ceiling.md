# EXP-0230 — M11-B raw int8 GEMM ceiling

## Question

How much headroom does a standard gfx906 int8 GEMM provide over the current
Q4_K B=4 projection mapping for the production FFN Down shape?

## Candidate

Raw `hipblasGemmEx` int8×int8→int32 GEMM with `M=32`, `N=5120`, and
`K=17408`, using one matrix operation for the full token chunk. The matrices
are deterministic synthetic int8 data; this is an arithmetic/dispatch ceiling
only, not a Q4_K implementation.

## Baseline

Eight native Q4_K `launch_q4k_wave_gemv_batched4` calls for the same
32-token/output shape, with Q8_1 quantization performed before timing.

## Environment

- GPU: AMD Instinct MI50, gfx906
- Model: `Qwen3.8-27B-Q4_K_M.gguf`
- Tensor shape: `[17408, 5120]`
- Build: `mi50-release`
- Benchmark: `MIINFER_INT8_GEMM_BENCH=1 build/mi50-release/miinfer-q4k-layout-bench /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf`

## Results

Interleaved samples, five operations per sample:

| Path | Batch time |
| --- | ---: |
| Native Q4_K, eight B4 launches | 2380.16 us |
| Raw hipBLAS int8 GEMM | 1716.00 us |

Raw int8 speedup: `1.38704x`.

## Interpretation

The measured arithmetic ceiling is materially below the roughly 2.2x
whole-prefill improvement required to move the qualified P513 layer-major
result from about 45.9 PP tok/s to 100 PP tok/s. It is also optimistic:
Q4_K cannot be passed directly to this GEMM because each 32-weight group has
row-dependent scale/minimum metadata. A correct implementation would need
additional groupwise processing, packing, and accumulation work.

This result does not prove that every custom packed GEMM is limited to 1.39x,
but it rules out treating generic int8 GEMM as the missing production path.

## Decision

REJECT as an implementation path; retain as an Amdahl ceiling measurement.

## Follow-up

Use this ceiling with the production stage profile to quantify the maximum
credible gain from quantized projections. Any further packed-GEMM experiment
must beat this bound after including Q4_K groupwise scale/minimum handling.
