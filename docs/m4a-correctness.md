# M4-A — Qwen3 execution correctness foundation

Status: `M4-A OPEN`

M4-A is being implemented as a correctness gate.  The current repository
contains the first primitive foundation, but it does not yet claim a complete
Qwen3 layer match.

## Pinned reference fixture

The reference is:

```text
milpster/gfx906-llama-cpp
6e4ef6c1a553b8f61ad77bba18e9ca05aa677295
```

The model is the pinned Qwen3-8B Q4_0 artifact with SHA256:

```text
458634762bea7dbe19f3ce0614465bafd15ee90e815229c427043afcf195d628
```

An explicit `hello` input produced reference token ID `14990`.  A temporary
debug-only callback in the separate reference checkout captured 28 layer-0
F32 checkpoints.  The raw fixture is retained outside this repository at:

```text
/home/fedora-workstation/Development/mi50-artifacts/m4a-trace-hello/
```

The capture is CPU reference execution (`-ngl 0`), not a performance result.
The temporary reference checkout was restored to its pinned clean revision
after capture.

Captured checkpoints, in execution order, include:

```text
embedding
attn RMSNorm and attention-normalized input
Q projection and Q norm
RoPE(Q)
V projection
K projection and K norm
RoPE(K)
attention output
attention residual
FFN RMSNorm
gate, up, SwiGLU, down
layer output
```

The exact dimensions and raw-file mapping are in the capture log:

```text
/home/fedora-workstation/Development/mi50-artifacts/m4a-reference-trace-hello.log
```

## Implemented foundation

`qwen3_primitives.hpp` and its implementation provide deterministic host
oracles for:

* FP16 conversion
* RMSNorm with FP32 accumulation
* adjacent-pair Qwen3 RoPE
* stabilized softmax
* SiLU/SwiGLU
* canonical Q4_0 dequantization and embedding lookup
* canonical Q6_K dequantization and GEMV

The gfx906 GPU probe provides correctness-first kernels for Q4_0 embedding
lookup, RMSNorm, and Q6_K GEMV.  It deliberately does not claim a complete
transformer layer or generation path.

Qwen3 uses Q/K tensors shaped as `[128, 32]` and `[128, 8]` after projection,
with 32 query heads and 8 KV heads.  The reference graph confirms Q/K norm is
applied after reshape and before RoPE, and that the Qwen3 FFN is
`SiLU(gate) * up` followed by the down projection.

## Current gate

Host primitive checks and physical-MI50 GPU primitive checks pass.  M4-A
remains open until MIInfer executes one complete layer and compares the
checkpoint outputs against the retained reference fixture with defined error
metrics.  No token-generation or end-to-end performance claim is made here.
