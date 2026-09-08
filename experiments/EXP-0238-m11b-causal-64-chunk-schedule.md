# EXP-0238 — M11-B causal 64-token prefill chunk

## Hypothesis

Staging a 64-token layer-major chunk before draining recurrent state and KV
writes in ordered B=4 groups will expose a larger causal token tile without
changing model semantics.

## Baseline

The qualified production path is EXP-0235: B=4 layer-major prefill, 46.22
tok/s at P513, with approximately 12.1 MiB of incremental recurrent-core
workspace. The 64-token build keeps the same B=4 projection kernels; it does
not yet implement a grouped quantized GEMM.

## Candidate

- Stage up to 64 normalized/Q8 inputs and QKV/gate projections.
- Run the existing causal recurrent/attention `run()` path in token order.
- Drain deferred tails in sixteen ordered B=4 groups.
- Allocate fixed 64-token per-layer workspaces only when layer-major prefill is
  enabled.
- `MIINFER_PREFILL_CHUNK=4` selects the old logical chunk for control;
  `MIINFER_PREFILL_CHUNK=64` selects the candidate.

## Environment

- GPU: AMD Instinct MI50, gfx906
- Model: `Qwen3.8-27B-Q4_K_M.gguf`
- Build: `mi50-release`
- ROCm: 6.4.0 / LLVM 20
- Prompt: 513 repeated `hello` tokens
- `MIINFER_PREFILL_LAYER_MAJOR=1`
- `MIINFER_HIP_GRAPH=0`
- generation disabled for throughput measurements

## Correctness

- Release build: PASS.
- Full CTest: **21/21 PASS**.
- P128/TG64 chunk-64 generation: completed without runtime error at 31.53
  tok/s decode.
- Chunk-4 and chunk-64 P128/TG64 runs completed with the same process output
  hash; the mixed stdout/stderr capture was not used as a numerical proof.

## Results

Interleaved same-binary P513 runs, three pairs:

| Chunk | Prefill ms | PP tok/s |
| ---: | ---: | ---: |
| 4 | 12345.57 | 41.55 |
| 64 | 11027.16 | 46.52 |
| 4 | 12298.19 | 41.71 |
| 64 | 10999.31 | 46.64 |
| 4 | 12274.38 | 41.79 |
| 64 | 11034.24 | 46.49 |

The chunk-64 median is 46.52 tok/s. The chunk-4 control is contaminated by
the candidate's fixed 64-token allocation footprint, so it is not a qualified
comparison to EXP-0235's smaller-workspace B=4 baseline. Against that baseline
the candidate is approximately neutral (+0.6%), within run-to-run variance.

## Interpretation

The schedule is causally valid and removes repeated layer-boundary turnover,
but repeating B=4 projection launches leaves the main token-parallelism
problem unsolved. The candidate also multiplies bounded per-layer staging
storage by 16; the larger allocation is not justified by a measured
production gain.

## Decision

**REJECT as a production default.** Keep the schedule shape and its correctness
checks as the integration point for a future true quantized skinny GEMM, but do
not enable the 64-token workspace or claim a throughput win.

## Follow-up

Implement and benchmark a native-layout token-tiled Q4_K/Q6_K kernel inside
this causal schedule. EXP-0231's B=64 microbenchmark is the lower-bound
reference; the end-to-end candidate must beat the qualified B=4 path after
workspace cost and correctness are included.
