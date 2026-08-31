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

## M4-B5 exact Q8 metadata propagation

The exact-sum implementation was then moved into the activation metadata
contract instead of recomputing the 32-lane Q8 sum inside every output-row
dot. `Q8ExactBlock` stores FP16 `d`, an exact `int16` lane sum, and the 32
int8 lanes in the same 36-byte footprint as `Q8_1Block`. The quantizer computes
the sum once per activation block, and the gfx906 dot kernels reuse it.

Focused real-shape Q4/Q8 tests pass for the exact-metadata kernel. In the
layer-6 teacher-forced discriminator, exact Gate correction reduces the layer
error from `11.1719` to `4.82812`; exact Up alone has no material effect, and
Gate+Up remains `4.82812`. For layer 1, exact V correction reduces the local
error from `0.0516891` to `0.0112991`; Q, K, and O alone do not materially
change that result. With exact metadata enabled for all Q4×Q8 projections,
teacher-forced layers 0–5 and 7–33 pass, while layers 6, 34, and 35 remain
outside the frozen bound (`4.82812`, `3.41504`, and `0.29126`).

The exact metadata path is now the default production Q4×Q8 contract. The
environment variable `MIINFER_EXACT_Q8_PROJECTIONS` remains as a controlled
replay selector; an empty value retains only the mandatory Down correction.
The old FP16-`s` kernels remain benchmark/diagnostic controls. The full
physical M4-B replay still fails intermediate layer gates, although the final
GPU logits are within the current `0.1` absolute bound (`0.0941614`) and the
argmax remains `8`. M4-B therefore remains open; no tolerance was widened and
generation work has not started.

The independent offloaded reference is also not numerically identical to the
CPU trace: the current MI50 comparison begins at `0.0521` on layer 0 and
reaches about `136` absolute error by layer 6. Therefore the external GPU
path does not yet justify replacing the CPU trace as the acceptance oracle.

## M4-B6 external layer-6 first-divergence trace

An independent internal trace was captured from the pinned CPU reference for
teacher-forced layer 6. The trace uses explicit token `14990` at position zero
and, importantly, captures `l_out-5` in the same run as the layer-6 tensors;
this avoids mixing the layer-6 input with the older full-forward fixture,
whose layer-5 output differs by up to `0.0786629`.

The tracked fixture is
[`tests/reference/qwen3/m4b-layer6/`](../tests/reference/qwen3/m4b-layer6/).
It contains the layer input plus attention, projection, normalization,
SwiGLU, Down, and final layer-output checkpoints. The temporary reference
worktree was pinned to commit
`6e4ef6c1a553b8f61ad77bba18e9ca05aa677295`; the model artifact is the pinned
Qwen3-8B Q4_0 file with SHA256
`458634762bea7dbe19f3ce0614465bafd15ee90e815229c427043afcf195d628`.

The diagnostic executable compares the same external trace against both the
host and MI50 implementations. Release and Debug produced the same results.
The host is exact or very close through `ffn_norm`, `gate`, and `up`; its
first strict checkpoint failure is `swiglu` at `max_abs=0.441895`. The large
absolute value is concentrated at a large activation (`5592.71` versus
`5592.27`), so the relative error there is about `0.008%`. Host
`ffn_output` and `layer_output` are at most `0.03125` from this same-run
external trace.

The GPU's first strict checkpoint failure is `q_projection`, at
`max_abs=0.0034914`; this is consistent with the known GPU projection
precision boundary rather than a newly discovered layer-layout error. The
GPU remains close through `ffn_norm`, `gate`, and `up`, but nonlinear
amplification produces `swiglu max_abs=1.44971`, followed by
`ffn_output max_abs=3.0918` and `layer_output max_abs=3.09375`. GPU and host
therefore remain mutually close in the early layer-6 stages, but do not yet
meet the external full-depth acceptance gate.

This result does not justify widening tolerances or changing production
semantics. It narrows the next investigation to the first GPU projection
precision boundary and the remaining host/reference numerical contract. The
external fixture comparator also includes a deliberate mutation discriminator
that must turn red when an expected checkpoint is changed.

M4-B6 conclusion: the independent layer-6 trace is captured and consumed
successfully, but M4-B remains open. No token-generation or performance work
was started.

## M4-B7 reference stability and exact-Q8 projection precision

The external capture method was repeated three times with identical pinned
CPU build, model, token `14990`, position, and environment. The shared
layer-6 checkpoints were bitwise-identical across runs. The earlier
`0.0786629` difference is therefore attributable to comparing different
fixture/capture inputs, not observed repeatability noise. The same-run
`layer-input.f32` remains the canonical teacher-forced input.

The precision probe was corrected to use `Q8ExactBlock` for all four input /
output combinations; it no longer exercises the retired FP16-`s` path. The
Release MI50 probe reported the following maximum absolute errors against the
external layer-6 projection checkpoints:

| Projection | F16 in/F16 out | F32 in/F16 out | F16 in/F32 out | F32 in/F32 out |
| --- | ---: | ---: | ---: | ---: |
| Q | 0.0034914 | 0.0034914 | 0.00121975 | 0.000002384 |
| K | 0.0008390 | 0.0008247 | 0.0007962 | 0.000000715 |
| V | 0.0008160 | 0.0002193 | 0.0008423 | 0.000000358 |
| Gate | 0.0095673 | 0.0095673 | 0.0053716 | 0.000011444 |
| Up | 0.0141068 | 0.0141068 | 0.0029894 | 0.000030518 |
| Down | 3.0918 | 3.0918 | 1.42871 | 0.0019531 |

The position-zero layer-6 fixture has one visible key/value entry, so its
attention probabilities are one. Q and K are therefore useful precision
canaries but cannot cause the final layer-output error in this fixture. The
causal path is V → attention output → O → residual → FFN. The existing
full-layer GPU result remains `layer_output max_abs=3.09375`.

The Q/V/Gate/Up/Down results show that retaining the GEMV result in F32 is the
strongest isolated correction: F32-input/F32-output is nearly exact for those
operations, while changing only the input boundary is generally ineffective.
The O row is now also derived correctly as `ffn_input - layer_input`: the host
derived O vector is only `0.00155926` from that external value, while MI50 is
`0.00305176` with F16 output and `0.00000190735` with F32 output. This makes
the O comparison usable for this position-zero diagnostic. The result is still
diagnostic evidence only; production precision has not been changed.

M4-B7 status: reference stability is established and the exact-Q8 precision
matrix is implemented. Q/K are precision canaries rather than causal sources
of the position-zero layer-output error because the attention prefix has one
entry. V, O, Gate, Up, and Down all become close to the external projection
trace with F32 output; Down changes from `3.0918` to `0.00195312`. This points
to output-F16 rounding as the minimum promising GPU correction, but no
production change is accepted yet. M4-B remains open; generation and
performance work remain out of scope.

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

## M4-B8 canonical oracle and precision policy

The full-depth fixture was recaptured from the pinned CPU reference in one
canonical configuration (`-ngl 0 -t 24 -tb 24`) using one explicit token
(`14990`). Two independent captures from the same build and settings were
byte-identical across all 39 F32 files. The previous fixture is preserved in
[`tests/reference/qwen3/m4b-single-token-legacy/`](../tests/reference/qwen3/m4b-single-token-legacy/)
because it was captured under different conditions; it is historical evidence
and is not the acceptance oracle.

Against the canonical fixture, host teacher-forced replay passes layers 0–34
and fails only layer 35 (`max_abs=1.18066`). This is the first remaining
host/reference blocker, so no GPU precision policy is accepted yet and no
tolerance was widened.

M4-B8 added diagnostic controls for testing projection boundaries without
changing the default production path:

```text
MIINFER_F32_INPUT_PROJECTIONS=v,o,gate,up,down
MIINFER_F32_OUTPUT_PROJECTIONS=v,o,gate,up,down
```

Each variable accepts a comma-separated projection list. The default remains
F32 → F16 → Q8Exact → GEMV → F16 → F32. A listed input uses direct F32 to
Q8Exact quantization; a listed output retains the GEMV result in F32. These
controls use the exact integer-sum Q8 metadata and are diagnostic only.

MI50 Release full-forward policy results against the canonical fixture were:

| Policy | Layer 6 | Layer 34 | Layer 35 | Final norm |
| --- | ---: | ---: | ---: | ---: |
| P0 current | 20.9287 | 16.8477 | 14.9043 | 0.06872 |
| P1 Down F32/F32 | 24.8223 | 24.7480 | 23.5283 | 0.09709 |
| P2 Gate/Up + Down | 10.3682 | 14.8184 | 18.2261 | 0.08081 |
| P3 V/O + FFN | 2.86621 | 5.84570 | 7.10254 | 0.04451 |
| P4 V/O/Gate/Up/Down | 1.50586 | 1.84180 | 2.19873 | 0.04395 |

Teacher-forced replay is the causal diagnostic: P3 reduces layer-6 error to
`0.0332451`, and P4 makes layers 6 and 34 pass at `0.03125` and `0.0198808`.
Layer 35 remains at `1.22998` with P4, consistent with the independent host
failure. The probes therefore show that F32 boundaries are decisive for the
GPU causal path, but they do not yet justify a production change or close
M4-B.

**M4-B8 status: OPEN.** The external oracle is canonical and repeatable; the
remaining layer-35 host/reference divergence must be localized before the
GPU precision policy is accepted.

## M4-B9 terminal layer-35 first-divergence trace

M4-B9 added an independent internal trace for terminal layer 35 using the
same pinned model, token (`14990`), CPU reference, and `-t 24 -tb 24`
configuration as M4-B8. The fixture is
[`tests/reference/qwen3/m4b-layer35/`](../tests/reference/qwen3/m4b-layer35/)
with F32 manifest hash
`af32ab6aabdb65b872fa129be8ef0e4c12173e6cd758703924f909f25f00a68b`.

Two independent captures using the same temporary reference instrumentation
were byte-identical for the selected internal tensors. The terminal boundary
was captured explicitly: `layer-output.f32` and `final-norm-input.f32` are
identical (`max_abs=0`). The external layer output is therefore exactly the
reference input to final RMSNorm.

The focused comparator reports external ↔ host, external ↔ MI50, and GPU ↔
host metrics, including max/mean absolute error, RMSE, relative error, worst
index/value, and first checkpoint outside the diagnostic `0.05` bound. Host
passes through `up`; its first strict failure is `swiglu` (`max_abs=0.105103`,
`max_rel=0.01077`), followed by `ffn_output=1.19824` and
`layer_output=1.18066`. The large absolute error is a small relative
nonlinear amplification, not evidence of an incorrect SiLU formula.

The unchanged MI50 path first fails at `ffn_input` (`0.101562`) and ends at
`layer_output=0.306641`. With diagnostic P4 F32 input/output boundaries,
`ffn_input` is `0.0166016`, while the shared tail remains
`swiglu=0.0950165`, `ffn_output=1.24683`, and `layer_output=1.22998`.
Debug and Release agree.

Hybrid external injection is causal evidence:

| Injected external values | GPU checkpoint | Max abs error |
| --- | --- | ---: |
| external gate + up | SwiGLU | `7.62939e-06` |
| external SwiGLU | Down projection | `0.000244141` |
| external Down output | residual | `0` |

Thus the terminal residual and GPU SwiGLU/down arithmetic are not the shared
blocker. The remaining issue is the reference-vs-host projection/nonlinear
numeric contract, amplified by SwiGLU. No tolerance was widened and no
production precision change was made.

**M4-B9 status: COMPLETE DIAGNOSTIC SLICE; M4-B remains OPEN.** The next
correction should address the shared host/reference numerical contract before
another GPU precision policy change.

## M4-B10 layer-35 Gate/Up projection isolation

M4-B10 added host-side hybrid SwiGLU diagnostics to separate the small Gate
and Up projection errors that are amplified by the multiplicative FFN
operation. The external layer-35 Gate and Up vectors produce a host SwiGLU
maximum error of `7.62939e-06`, so the host SiLU/multiply implementation is
not the source of the failure.

The pinned external layer-35 SwiGLU comparison was:

| Gate input | Up input | Max abs vs external |
| --- | --- | ---: |
| external | external | `7.62939e-06` |
| host | external | `0.104843` |
| external | host | `0.0908432` |
| host | host | `0.105103` |

Both host projections contribute to the discrepancy, with Gate the larger
single-source contribution for this fixture. At the worst host SwiGLU error
index (`5607`), the Gate difference is `0.000995636` and the Up difference is
`0.0000991821`; the large SwiGLU absolute error is therefore nonlinear
amplification of small projection differences. GPU hybrid injection continues
to show that GPU SwiGLU, Down, and residual arithmetic are independently
accurate when supplied with external inputs.

This does not yet identify whether the shared Gate/Up projection difference is
activation quantization or CPU accumulation order. The next diagnostic should
compare the pinned Q8 quantization contract and then instrument the implicated
projection's block accumulation. No production precision or tolerance change
was made.

**M4-B10 status: COMPLETE DIAGNOSTIC SLICE; M4-B remains OPEN.**
