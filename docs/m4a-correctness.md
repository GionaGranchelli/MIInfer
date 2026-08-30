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

Host primitive checks and physical-MI50 GPU primitive checks pass.  The
correctness-only host executor now composes the complete layer-0 path for the
single-token position-zero fixture and compares all 28 checkpoints.  It passes
with the frozen stage-specific tolerances below; the observed final-layer
maximum absolute difference is approximately `7.6e-3`, caused primarily by
Q8 activation quantization and accumulation-order differences.

The comparator reports maximum/mean absolute error, meaningful relative error
(reference magnitudes below `1e-2` use the absolute gate), RMSE, maximum-error
index, and the reference/actual values at that index.  A deliberate mutation
test is included and is detected by the comparator.

The currently frozen host comparison policy is:

| Stage | Absolute tolerance | Relative tolerance |
|---|---:|---:|
| embedding, attention RMSNorm/weight | `2e-4` | `2e-4` |
| Q/K/V projections and reshapes | `2e-4` | `1e-2` |
| Q/K RMSNorm | `5e-3` | `1e-1` |
| Q/K weighted norm, RoPE, views | `2.5e-2` | `1.5e-1` |
| attention through FFN input | `2e-2` | `5e-1` |
| FFN weighted norm/SwiGLU | `2e-2` | `1.5e-1` |
| FFN output/final residual | `2e-2` | `4e-1` |

These tolerances are not a performance escape hatch: the host run's observed
errors are printed in full, and the mutation test verifies that a material
checkpoint change fails.

## M4-A3 GPU composition acceptance

The correctness-first MI50 executor now composes the same layer-0 path on the
physical GPU.  It uses the M3 static model plan and the accepted M2 projection
kernels, while deliberately copying intermediate vectors back for diagnosis.
The GPU test compares every canonical checkpoint against both:

```text
the host layer-0 executor
the pinned reference trace
```

The additional attention-score and attention-probability diagnostics are also
checked against the host executor.  The accepted physical runs passed in both
MI50 configurations:

```text
mi50-debug   28/28 checkpoints, host/reference parity: PASS
mi50-release 28/28 checkpoints, host/reference parity: PASS
```

The position-zero acceptance commands are:

```bash
./build/mi50-debug/miinfer-qwen3-layer0-gpu-test \
  /path/to/Qwen3-8B-q4_0-b968826d.gguf \
  /path/to/m4a-trace-hello

./build/mi50-release/miinfer-qwen3-layer0-gpu-test \
  /path/to/Qwen3-8B-q4_0-b968826d.gguf \
  /path/to/m4a-trace-hello
```

The observed attention-score maximum absolute difference was approximately
`9.55e-3`; probabilities were identical for the one-token position-zero
fixture.  All compared vectors were finite.  A GPU-specific test-only
`swap-gate-up` mutation was detected at `swiglu` and downstream checkpoints,
demonstrating that the composition gate is sensitive to semantic breakage.

The position-zero executor remains a compatibility wrapper around the
stateful implementation.  Multi-position execution is covered separately
below; no token-generation or end-to-end performance claim is made here.

## M4-A4 KV-cache contract

M4-A4 uses a single-sequence, layer-0 cache with capacity fixed by the test
fixture.  Host and GPU storage have the same logical layout:

```text
[kv_head][position][head_dim]
```

Keys are stored after position-specific RoPE; values are stored unchanged.
Only positions `[0, length)` are valid, and an append must use exactly the
current `length`.  Reset clears storage and returns `length` to zero.  The
position-zero executor remains available as a compatibility wrapper, while
the stateful overloads own the four-position correctness test.

The deterministic M4-A4 sequence is:

```text
[14990, 42, 31415, 2718]
```

The host and MI50 GPU tests execute positions `0..3` and require cache length
`1..4`, preservation of every earlier entry, current K/V agreement, causal
attention vectors of exactly `heads * (position + 1)` elements, finite
probabilities summing to one, and deterministic results after reset.  The
physical Release and Debug runs pass these checks.  Test-only mutations for
wrong RoPE position, ignoring earlier cache entries, and corrupting a prior
cache entry are all detected; out-of-order append is rejected.

The stateful GPU test currently uses the validated host executor as its
position `1..3` numerical authority.  The retained pinned external trace is
still position-zero-only, so a four-step external callback trace remains the
last evidence item before M4-A can close.

## Host layer-0 acceptance run

```bash
./build/host-only/miinfer-qwen3-layer0-test \
  /path/to/Qwen3-8B-q4_0-b968826d.gguf \
  /path/to/m4a-trace-hello 14990
```

The current real-artifact run reports `layer-0 host comparison: PASS` for:

```text
embedding, attention RMSNorm/weight, Q/K/V projections and reshapes,
Q/K RMSNorm/weights/RoPE, GQA attention output, attention residual,
FFN RMSNorm/weight, gate, up, SwiGLU, down, and final layer residual.
```

The CTest mutation gate ensures that a changed checkpoint cannot be accepted
because the comparator is accidentally permissive.
