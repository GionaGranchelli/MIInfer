# M4-B — Full Qwen3 single-token forward

Status: `OPEN`

M4-B is the full-depth correctness gate. It executes one explicit token at
position zero through all 36 Qwen3 layers, final RMSNorm, and the Q6_K output
projection. Sampling and token generation are out of scope.

## Independent reference

The tracked fixture in
[`tests/reference/qwen3/m4b-single-token/`](../tests/reference/qwen3/m4b-single-token/)
was captured from the pinned `milpster/gfx906-llama-cpp` reference at commit
`6e4ef6c1a553b8f61ad77bba18e9ca05aa677295`, using the pinned Qwen3-8B Q4_0
artifact and explicit token `14990`. It contains the embedding, all
`l_out-0` through `l_out-35` vectors, final normalized hidden state, and full
vocabulary logits.

The reference capture uses `-ngl 0` so the host comparison is against the
reference CPU execution path. A separate offloaded reference capture is
retained outside the repository for diagnosing GPU arithmetic differences; it
is not mixed into the host acceptance fixture.

## Implemented slice

`execute_qwen3_forward_host()` now reuses the explicit layer implementation
for every planned layer, with one position-zero cache per layer, then applies
the final norm and the pinned reference's Q6_K × Q8_K output path. The MI50
path has the corresponding correctness-first executor and retains per-layer
outputs for comparison.

Both paths complete all 36 layers without NaN/Inf on the real artifact.

## Current evidence

The host and GPU executors are not yet accepted against the full-depth oracle.
Layer 0 remains close, but depth-dependent differences accumulate beyond the
currently frozen per-layer bounds. The Q6_K decoder required corrected
128-value-half and nibble-selection indexing; after those fixes, host and
MI50 select the reference argmax (`8`), and host full logits pass the current
0.1 absolute bound. MI50 logits remain just outside that bound (`0.161`),
while several intermediate layer outputs also fail. This remains an open
correctness blocker, not a reason to broaden tolerances.

The remaining distinction to resolve is the numerical projection contract:
the layer-0 runtime path uses Q8_1 activation quantization before Q4_0 GEMV,
while the `-ngl 0` reference uses its CPU Q4_0 × Q8_0 path. The output path
uses Q6_K × Q8_K in the reference, so that contract is mirrored separately.
The separate offloaded capture helps distinguish numerical drift from a
semantic layer-indexing or buffer-reuse error.

## M4-B2 depth-drift diagnostics

The correctness-only forward executables now support teacher-forced replay:

```text
reference layer[N-1] → MIInfer layer[N] → compare reference layer[N]
```

The initial replay set is layers 1, 2, and 6; `--teacher-forced-all` runs all
36 layers. This separates a local layer defect from error propagated by the
free-running stack. The host replay uses a GGML-compatible round-to-nearest
FP16 conversion for Q8 scale metadata, while the normal host forward remains
unchanged for accepted-result continuity.

On the pinned CPU trace, host teacher-forced replay passes layers 1–5 and
7–33, while layer 6 is just outside the existing absolute bound
(`max_abs=0.103516`) and layers 34–35 are also outside it. This is evidence
that the current `0.05` bound is not a sufficient full-depth acceptance
criterion, but it is not a reason to widen that bound yet.

MI50 teacher-forced replay is close at layer 2 (`max_abs=0.00270`) but fails
layer 1 (`0.05169`) and layer 6 (`11.1719`) against the CPU trace. GPU-vs-host
stage diagnostics localize the layer-6 GPU difference to the nonlinear FFN
tail: the GPU and host gate/up values are close, while SwiGLU and the
following down projection amplify the difference. This remains a numerical
contract investigation, not a proven semantic failure.

The independent offloaded reference is also not numerically identical to the
CPU trace: the current MI50 comparison begins at `0.0521` on layer 0 and
reaches about `136` absolute error by layer 6. Therefore the external GPU
path does not yet justify replacing the CPU trace as the acceptance oracle.

## Physical acceptance command

The default CTest entries intentionally remain artifact-free unit/regression
tests and may skip the real-model forward comparison when no paths are
provided. The non-vacuous physical gate is:

```bash
scripts/run-m4b-acceptance.sh \
  /path/to/Qwen3-8B-q4_0-b968826d.gguf \
  tests/reference/qwen3/m4b-single-token
```

It fails if the model, required checkpoints, or MI50 Release binaries are
missing, and it propagates a real host or GPU comparison failure.

## Acceptance target

The gate remains:

```text
independent reference ↔ host ↔ MI50 Debug ↔ MI50 Release
```

for embedding, every layer output, final norm, and full logits, with explicit
layer-localized diagnostics. M4-B remains open until the numerical contract
and final-logit/argmax comparison are resolved.
