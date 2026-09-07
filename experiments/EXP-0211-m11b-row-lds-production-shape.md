# EXP-0211 — M11-B Row-LDS B=8 Production-Shape Rejection

## Hypothesis

One output row per 8-wave workgroup, with that row's Q4/Q5/Q6 weights staged in
LDS, can reuse weights across eight prompt vectors and improve the B=4
layer-major prefill path.

## Baseline

The qualified M11-B candidate is the opt-in layer-major path using validated
B=4 native projection and paired-SwiGLU kernels. On the qualification model,
its three-run P512 median is 12,812.97 ms, or 39.96 prompt tokens/s.

## Candidate

Temporary row-LDS kernels were implemented for Q4_K, Q5_K, Q6_K, and paired
Q4_K SwiGLU. Each workgroup owned one output row, staged only that row's
weights, and assigned one Wave64 to each of eight input vectors. The runtime
was switched to B=8 only under `MIINFER_PREFILL_LAYER_MAJOR=1`.

## Environment

- GPU: gfx906 MI50/MI60-visible device
- SCLK: 1606 MHz
- HBM/MCLK: 1000 MHz
- ROCm: 6.4.0 / LLVM 20
- Model: `Qwen3.8-27B-Q4_K_M.gguf`
- Graph capture: disabled for prefill comparison

## Correctness

The independent Q4_K row-LDS check was bit-exact against two B=4 launches at
64 rows × 17,408 columns. The B=8 production smoke check also selected the
same `hello` continuation as the B=4/default path for P16 and 16 generated
tokens.

## Results

The small-row result was misleading:

| Shape | Row-LDS B=8 | Two B=4 | Relative |
| --- | ---: | ---: | ---: |
| 64 × 17,408 | 0.01923 ms | 0.03732 ms | 1.94× |
| 5,120 × 17,408 | 1.60531 ms | 0.59463 ms | 0.37× |

The production-shaped runtime smoke result was 24.88 P128 tok/s, versus the
qualified B=4 layer-major result of 41.21 P128 tok/s. The one-row workgroup
launch count scales with output rows and overwhelms the per-row weight reuse.

## Decision

**REJECT.** Remove the temporary kernels and B=8 runtime wiring. Retain the
negative result: row-LDS reuse is not a production prefill strategy for these
skinny projections at model-sized row counts.

## Follow-up

The next valid M11-B projection experiment must preserve the B=4 workgroup
row grouping while increasing useful input reuse, or implement a true
production-shaped skinny GEMM. Do not infer end-to-end value from small-row
LDS microbenchmarks.
