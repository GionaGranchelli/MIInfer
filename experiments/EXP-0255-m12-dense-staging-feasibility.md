# EXP-0255 — M12 dense staging feasibility

## Hypothesis

One whole-matrix Q4_K → FP16 expansion followed by a real hipBLAS GEMM can
break the current packed-Q4 B=4 projection path when enough prompt tokens are
available to amortize staging.

## Motivation

M11-B is frozen at the qualified 46.22 P513 tok/s result. The dominant
recurrent deferred tail is matrix-shaped, while the current packed-Q4
executor has an optimistic whole-path ceiling of roughly 63–64 tok/s. This is
the first M12 test of a separate matrix-oriented prefill primitive.

## Baseline

The exact current FFN-Down production shape uses the existing gfx906 native
Q4_K wave kernel in repeated batched-B4 launches:

```text
rows = 5120
columns = 17408
tensor = blk.8.ffn_down.weight
```

The measured control is the existing `launch_q4k_wave_gemv_batched4` path,
repeated until the requested batch is consumed. It excludes activation
quantization, matching the candidate's exclusion of input materialization.

## Candidate

- Expand the canonical GGUF Q4_K matrix once into row-major FP16 storage.
- Reuse the 178,257,920-byte staging buffer for all measured batches.
- Run `hipblasGemmEx` with FP32 accumulation and FP16 output.
- Charge the one-time expansion to every candidate measurement:
  `repack_us + GEMM_us`.
- Do not change the production executor or decode path.

The temporary lab primitive is `launch_m12_q4k_to_fp16`; it uses one 256-thread
workgroup per Q4_K block and is retained only as a feasibility benchmark.

## Environment

- GPU: AMD Instinct MI50 / gfx906
- Model: `/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf`
- Build: `mi50-release`, gfx906, Release
- ROCm: 6.4.0 / LLVM 20
- Baseline production commit: `556a82a`
- Exact tensor: `blk.8.ffn_down.weight`, Q4_K `[17408, 5120]`

## Correctness

- Q4_K → FP16 sampled rows: max absolute error `1.52588e-5`.
- FP16 GEMM sampled outputs: max absolute error `1.44149`; accepted under the
  FP16-output check of absolute error `2.0` or relative error `1%`.
- No model/runtime integration or generation correctness claim was made.

## Results

Repack median: `1097.92 us`.

| Batch | Existing B4 | GEMM | Repack + GEMM | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 64  | 4736.48 us | 2374.88 us | 3472.80 us | 1.364× |
| 128 | 9458.08 us | 2374.40 us | 3472.32 us | 2.724× |
| 256 | 18973.91 us | 3643.84 us | 4741.76 us | 4.001× |
| 512 | 38144.94 us | 5471.84 us | 6569.76 us | 5.806× |

## Interpretation

The matrix primitive breaks the current-family projection ceiling once at
least 128 token activations can share the staged matrix. At the current causal
B64 chunk it reaches only `1.364×`, below the M12 feasibility kill gate of
approximately `2×`. Dense staging is not a drop-in replacement for the
existing M11-B schedule, but it remains a credible backend component if a
separate chunkwise Gated DeltaNet schedule can expose B128–B512 tiles.

## Decision

**RETAIN AS M12 RESEARCH PRIMITIVE; DO NOT PROMOTE TO PRODUCTION.** The
primitive earns a second experiment only in combination with a chunkwise
recurrent dataflow. Do not add runtime staging, persistent model expansion, or
decode changes based on this result alone.

## Follow-up

Prototype one recurrent-layer chunkwise/WY Gated DeltaNet oracle at chunk size
64–128. Require exact state/output agreement with the current token-recurrent
oracle before combining it with the B128+ dense projection path.
