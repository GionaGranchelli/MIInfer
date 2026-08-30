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
free-running stack. The host correctness path now uses the validated GGML-
compatible round-to-nearest FP16 conversion for Q8 scale metadata everywhere.

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

## M4-B3 precision isolation

The layer-6 probes now separate the two suspected projection boundaries while
keeping the kernel geometry and quantized weights fixed. Relative to the
reference-conditioned host trace, the four GPU paths produced:

| Path | Gate max abs | Up max abs | SwiGLU max abs | Down max abs | Layer max abs |
| --- | ---: | ---: | ---: | ---: | ---: |
| F16 input → Q8_1 → F16 output | 0.03496 | 0.02287 | 1.09277 | 11.2754 | 11.2754 |
| F32 input → Q8_1 → F16 output | 0.02754 | 0.02287 | 3.78613 | 4.72461 | 4.72461 |
| F16 input → Q8_1 → F32 output | 0.01273 | 0.00467 | 0.68750 | 1.87402 | 1.87402 |
| F32 input → Q8_1 → F32 output | 0.01115 | 0.00365 | 0.45410 | 1.83203 | 1.83203 |

Hybrid injection exonerates the GPU SwiGLU arithmetic: reference gate/up
inputs produce a SwiGLU error of `5.96e-8`. Injecting reference SwiGLU into
the GPU down projection still leaves `3.27539` max absolute error, so the
projection result boundary and/or down-projection input quantization remains a
real local contributor. The best diagnostic path reduces the layer error from
`11.2754` to `1.83203`, but does not meet the frozen `0.05` acceptance bound.

The Q8 block probe confirms that the pre-quantization boundary changes the
actual quantized input: 63 of 128 blocks differ, 36 int8 lanes differ, and
the largest stored-scale difference is `6.10352e-05` between F16-input and
direct-F32-input quantization. These are diagnostic variants only; the
accepted production path has not been changed.

The host teacher-forced result after making round-to-nearest FP16 conversion
canonical is: layers 0–5 and 7–33 pass, while layers 6, 34, and 35 remain
outside the frozen bound. Therefore M4-B remains open. No tolerance widening
or token-generation work is justified yet.

## M4-B4 exact down-projection contract

The diagnostic comparison used the pinned layer-6 SwiGLU vector as input to
the down projection and emitted F32 results. With the production F16-input Q8
blocks, the current FP16 `s` correction has `max_abs=0.461914`, while both the
exact integer-sum dot4 path and the direct signed-Q4 GPU oracle have
`max_abs=0.03125`, `mean_abs=0.00062987`, and `RMSE=0.000964196` against the
reference-conditioned host down output. The exact dot4 and direct signed
oracles are identical in this comparison (`max_abs=0`).

Using direct-F32-input Q8 quantization reduces the exact/direct error further
to `max_abs=0.0117188`, but the current FP16-`s` path remains about `0.464844`.
This separates the dominant error: the packed dot arithmetic is correct, and
the lossy FP16-scaled sum used for zero-point correction is the local defect.

For the worst row (`2276`), the block analysis reports total absolute
correction error `1.93343` and signed accumulated correction error `0.43165`.
The largest individual examples include block 266 (`q8_sum=-126`, stored
`s=-1312`, exact `d*sum=-1311.19`) and block 13 (`q8_sum=127`, stored
`s=1082`, exact `d*sum=1082.48`).

The exact/direct probe currently uses the host layer trace's pinned
reference-conditioned SwiGLU and down output; the retained external offloaded
trace contains layer outputs rather than internal layer-6 FFN tensors. The
minimum production correction is now integrated only for the Down projection:
it uses exact integer Q8 lane sums while retaining the canonical Q8_1 storage
and FP16 GEMV output. Q/K/V/O/gate/up continue to use the accepted FP16-`s`
control. The old FP16-`s` path remains available as a diagnostic control.
The isolated Down contract meets the frozen layer-local bound, but applying it
to the full layer does not close the model-level layer-6 failure: upstream
gate/up precision differences still enter SwiGLU and are amplified by Down.
No tolerance has been widened and M4-B remains open.

The post-integration Release replay confirms the scope of that result. The
Down-only correction does not alter the earlier gate/up path: the layer-0 run
still reports a gate-vs-host mismatch even though its external-reference
comparison remains within the layer-0 gate. The external-reference comparison
passes layer 1 before later depth drift, while the full GPU replay still fails
the layer-6 and later layer-output gates. The isolated Down-contract
improvement must therefore not be described as a complete model-level fix.

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
