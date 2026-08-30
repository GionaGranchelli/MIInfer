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

## Acceptance target

The gate remains:

```text
independent reference ↔ host ↔ MI50 Debug ↔ MI50 Release
```

for embedding, every layer output, final norm, and full logits, with explicit
layer-localized diagnostics. M4-B remains open until the numerical contract
and final-logit/argmax comparison are resolved.
