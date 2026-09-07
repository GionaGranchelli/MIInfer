# EXP-0207 — B=8 LDS Weight-Reuse Prefill

## Hypothesis

An eight-token, multi-wave Q4_K/Q6_K GEMV that stages four output rows of
weights in LDS can amortize weight traffic without the register pressure of a
single-wave B=8 kernel.

## Baseline

The validated layer-major production candidate uses B=4 native projection and
paired SwiGLU kernels. On `Qwen3.8-27B-Q4_K_M.gguf` it measured about 39--40
tok/s at P512, with the path still opt-in because its long-generation
correctness is unresolved.

## Candidate

Temporary gfx906 kernels were added for:

- Q4_K GEMV, two waves per output row and four rows per workgroup;
- Q6_K GEMV with the same LDS layout;
- paired Q4_K gate/up SwiGLU;
- eight prompt vectors processed in four two-token groups.

The pipeline was temporarily switched to B=8 for an isolated P16 check.

## Environment

- GPU: gfx906 (MI50/MI60-visible device)
- ROCm: 6.4.0 / LLVM 20
- Model: `Qwen3.8-27B-Q4_K_M.gguf`
- Workload: layer-major prefill, P16

## Results

The first one-row-per-workgroup mapping produced a different selected token
(`hero` rather than the B=4 path's `hello`) and was discarded. The revised
four-row/two-wave mapping then caused a GPU memory access fault during P16.
No performance number is qualified from the faulting candidate.

The generated 5.1 GB diagnostic core was removed. No B=8 kernel or public
declaration is retained in the production tree.

## Decision

**REJECT.** The candidate failed the minimum standalone correctness bar and
triggered a device fault before a valid benchmark could be established.

## Follow-up

If B>4 is revisited, validate each kernel against an independent GPU-side
reference with tiny buffers before wiring it into the model pipeline. Do not
use a production model run as the first correctness test for a multi-wave LDS
mapping.
