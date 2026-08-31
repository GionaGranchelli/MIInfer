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

The host and GPU executors remain under full-depth acceptance. The Q6_K
decoder required corrected 128-value-half and nibble-selection indexing;
after those fixes, both paths select the reference argmax (`8`). The MI50
full-forward path passes its canonical layer/logit gate under the accepted
precision policy. Host full-forward still has a depth-composition failure,
with the first strict failure at layer 2 (`max_abs=0.117966`); this remains an
open correctness blocker, not a reason to broaden tolerances.

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

## M4-B11 Q8 identity and CPU accumulation contract

M4-B11 tested whether the layer-35 Gate/Up discrepancy was introduced by Q8
activation quantization or by the host Q4×Q8 row accumulation. The test
replays both MIInfer's current formula and the pinned x86 AVX contract on the
exact external layer-35 `ffn_norm` vector. The pinned implementation computes
the quantizer inverse as `127 / amax` and rounds to nearest-even; MIInfer
computes `1 / (amax / 127)` and uses `std::round`.

For all 128 activation blocks these contracts produced identical stored FP16
scales and identical int8 lanes:

```text
different blocks: 0
different lanes:  0
scale-bit delta:  0
```

Using the exact external `ffn_norm` as input, the current MIInfer sequential
Q4×Q8 replay matched the external projection checkpoints:

| Projection | Max abs | Mean abs | RMSE |
| --- | ---: | ---: | ---: |
| Gate | `7.62939e-06` | `1.22185e-07` | `3.48546e-07` |
| Up | `1.52588e-05` | `1.27735e-07` | `6.08251e-07` |

Critical-row double and four-accumulator variants did not improve these
already-small errors. This rules out a production Q8 metadata change or a
general Gate/Up accumulation-order correction as the next fix. The original
host Gate/Up differences arise because the normal host layer path consumes
its own slightly different `ffn_norm`; at the host-vs-external SwiGLU worst
index, Gate differs by `0.000995636` and Up by `0.0000991821`.

The Q8 comparison is a source-compatible replay of the pinned x86 AVX
quantizer rather than a new multi-gigabyte runtime-block artifact. It is
therefore recorded as contract evidence, not as a claim that the external
temporary checkout was modified or that its private transient blocks were
persisted. No production behavior or tolerance changed.

**M4-B11 status: COMPLETE DIAGNOSTIC SLICE; M4-B remains OPEN.** The next
investigation should locate the earlier FFN-input/attention-output numeric
contract that produces the differing `ffn_norm`, beginning with the external
conditioned O projection and residual path.

## M4-B12 pre-FFN residual and RMSNorm isolation

M4-B12 replayed the layer-35 pre-FFN path with the external attention output
and external layer input. The host O projection replay was close to the
derived external O tensor (`max_abs=9.15527e-05`), and adding external O to
the external layer input reproduced the external `ffn_input` exactly. Using
the replayed O plus the external layer input produced only `3.05176e-05`
maximum error. This makes the residual add exact and leaves only the earlier
attention-output-to-O path as a possible source of the normal-path
`ffn_input` difference.

Four RMSNorm reductions were evaluated from the exact external `ffn_input`:
float sequential, MIInfer's double-product accumulation, the pinned ggml
contract (float product widened into a double accumulator), and four float
accumulators. The pinned CPU implementation was inspected directly; its
`ggml_float` is `double`, but its source expression computes `x[i] * x[i]`
before widening. All candidates drove the existing Gate/Up → SwiGLU → Down →
residual tail to the same small result:

```text
layer-output max_abs: 0.000488281
layer-output mean_abs: 4.7646e-06
layer-output RMSE: 1.07066e-05
```

The double and pinned-ggml variants produced an exact external `ffn_norm`
vector for this fixture. RMSNorm reduction order is therefore not the shared
layer-35 blocker. The remaining normal-path difference enters before the
external-conditioned O replay, in the attention-output path that feeds O.
No production behavior, precision policy, or tolerance changed.

**M4-B12 status: COMPLETE DIAGNOSTIC SLICE; M4-B remains OPEN.** The next
investigation should isolate the attention-output-to-O input contract rather
than changing RMSNorm or the FFN tail.

## M4-B13 attention RMSNorm, V, and position-zero GQA isolation

M4-B13 used the position-zero identity to remove generic attention mechanics
from the layer-35 investigation. Host expanded V matched host attention output
exactly, and replaying V from either the external or host `attn_norm` matched
the external V projection to `4.76837e-07`. The V projection and GQA mapping
are therefore not the source of the large normal-path error.

The external trace exposed the remaining boundary: external attention output
matches `FP16(expanded V)` exactly, while the host retains the expanded value
in F32. The unrounded replay produced the known layer-35 error of `1.18066`;
materializing the replayed attention output through FP16 reduced the complete
V → GQA → O → residual → FFN tail to `0.000488281` maximum error. This is a
reference representation/precision contract, not a Q/K/score/softmax issue
for the single visible KV entry at position zero.

Float, current double, literal pinned-ggml, and four-float attention RMSNorm
reductions were also replayed from the external layer input. The double and
pinned-ggml variants matched external `attn_norm` exactly, but all unrounded
variants retained the same layer error; with FP16 attention-output
materialization they all reduced it to `0.000488281`. No production behavior,
precision policy, or tolerance changed.

**M4-B13 status: COMPLETE DIAGNOSTIC SLICE; M4-B remains OPEN.** The next
investigation should test the minimum production policy for the attention
output FP16 boundary before O projection.

## M4-B14 production attention-output FP16 boundary

M4-B14 applied the B13 precision contract to production host and MI50 layer
execution: attention is accumulated in F32, materialized through round-to-
nearest FP16, then returned to the existing F32 interface before O
projection. No other projection precision boundary or tolerance changed.

The focused external layer-35 host comparison now passes through the terminal
layer output with `max_abs=0.000488281`. The full MI50 Release forward also
passes its layer/logit gate and retains `argmax=8`. The real acceptance still
fails overall because the host full-forward path has earlier depth drift; its
first strict failure is layer 2 (`max_abs=0.117966`), while final host logits
remain outside the current gate (`max_abs=0.11144`). This is not evidence to
widen tolerances. The production boundary is retained because it fixes the
identified layer-35 contract and improves earlier host layers, but M4-B is
not closed.

Both Debug and Release builds succeeded and both standard CTest suites passed
16/16. No generation or performance work was introduced.

**M4-B14 status: KEEP; M4-B remains OPEN.** The next investigation should
localize the remaining host full-forward depth drift beginning at layer 2,
while preserving the accepted attention-output FP16 boundary.

## M4-B15 host sequential-composition localization

M4-B15 added a test-only composition diagnostic to both forward executables.
It reconstructs layers 0–2 by chaining the existing complete
teacher-forced-layer API, then compares each chain against a
reference-conditioned isolated replay and the canonical external per-layer
outputs. It also compares the reconstructed chain with the existing
full-forward entry points, distinguishing orchestration defects from ordinary
state propagation.

On the pinned artifact, the host results were:

| Layer | Isolated vs external | Sequential vs external |
| ---: | ---: | ---: |
| 0 | `0.00549483` | `0.00549483` |
| 1 | `0.000249505` | `0.0110674` |
| 2 | `3.8147e-06` | `0.117966` |

The first nonzero sequential-vs-isolated difference is layer 1 input, exactly
the layer-0 output difference. The full host forward is bitwise identical to
the reconstructed sequential chain through layer 2. Therefore layer 2 is
the first threshold crossing, not evidence of a layer-2-only orchestration
bug; host drift is inherited through sequential composition. The MI50 probe
showed the same first inherited divergence at layer 1 input, while its
reference-conditioned layer outputs stayed within the current bound through
layer 2. Both full-forward entry points were bitwise identical to their
reconstructed chains.

The diagnostic does not change production behavior or tolerances. It is
invoked explicitly with:

```text
miinfer-qwen3-forward-test MODEL TRACE --composition-diagnostic
miinfer-qwen3-forward-gpu-test MODEL TRACE --composition-diagnostic
```

**M4-B15 status: COMPLETE DIAGNOSTIC SLICE; M4-B remains OPEN.** The evidence
does not support changing layer-2 arithmetic or adding a host buffer fix.

## M4-B16 layer-0 first-divergence and materialization probe

M4-B16 reused the independent M4-A5 position-zero fixture
(`m4a4-four-position/pos-0-*.f32`) to compare all 28 host layer-0
checkpoints. The first nonzero mismatch is `q_projection`:

```text
max_abs=4.75142e-05
```

For the position-zero causal path, Q/K cannot affect the result because the
single visible key produces a softmax weight of one. The first causal
projection difference is therefore the small V mismatch (`3.50252e-05`),
followed by attention output (`1.22070e-04`) and final layer output
(`5.48154e-03`). The layer-0 output error is gradual; there is no material
single operation jump or evidence of a layer-0 state-management defect.

The cheap proposed output-boundary discriminator was negative:

| Layer-0 output representation | Max abs vs external |
| --- | ---: |
| Current F32 result | `0.00548154` |
| Additional FP16 round-trip | `0.00606418` |

An extra FP16 round-trip at the layer output would therefore make this
fixture slightly worse and is not a production correction. The existing
layer-0 host/GPU comparator was also extended to consume the `pos-0-*`
fixture. It reports host layer output `0.00548154`; the GPU run remains
within its established stage bounds except the stricter `ffn_rms` diagnostic
threshold (`gpu-vs-reference max_abs=0.0498233`).

Release and Debug builds pass, and both standard CTest suites remain `16/16`.
No production semantics or tolerance changed.

**M4-B16 status: COMPLETE DIAGNOSTIC SLICE; M4-B remains OPEN.** The final
layer-output FP16 hypothesis is rejected. The next investigation should
follow the causal V/FFN precision path inside layer 0 rather than add a
global inter-layer cast.

## M4-B17 layer-0 causal attention-output replay

M4-B17 added a correctness-only attention-output override to the host and
MI50 layer probes. At position zero, where the sole attention probability is
one, the external attention output is the GQA-expanded V. Replacing the
computed attention output with that external tensor isolates the downstream
O/residual/FFN path without changing production execution.

The host result was decisive:

| Layer-0 path | Max abs vs external layer output |
| --- | ---: |
| Normal attention output | `0.00548154` |
| External attention injected | `1.90735e-06` |

The host injected tail is effectively exact: its `ffn_input` error is
`3.57628e-07`, `ffn_norm` is `1.49e-07`, and the final residual is
`1.90735e-06`. This exonerates O, residual, FFN normalization, Gate/Up,
SwiGLU, and Down as independent sources of the layer-0 drift. The normal
path's causal input is the small V/attention-output difference identified in
M4-B16.

The MI50 result shows the same causality, with a smaller but nonzero GPU tail:

| Layer-0 path | Max abs vs external layer output |
| --- | ---: |
| Normal attention output | `0.00704432` |
| External attention injected | `0.00282186` |

Thus the attention/V perturbation accounts for most of the MI50 layer-0
error, while the remaining `~0.0028` is GPU downstream numerical variance.
The override is diagnostic-only; no production arithmetic, tolerance, or
runtime policy changed.

Release probes built and ran successfully. **M4-B17 status: COMPLETE
DIAGNOSTIC SLICE; M4-B remains OPEN.** The next investigation should decide
whether the remaining causal V projection difference is a missing reference
precision boundary or acceptable cross-backend projection variance.

## M4-B18 V precision and causal replay

M4-B18 reused the canonical layer-35 external trace and tested the four
exact-Q8 V projection policies. The host projection, conditioned on the
external attention-normalized input, remained effectively exact:

```text
max_abs=4.76837e-07
```

The MI50 local V results were:

| Policy | V max abs vs external |
| --- | ---: |
| F16 -> Q8Exact -> F16 | `0.00290415` |
| F32 -> Q8Exact -> F16 | `0.00161266` |
| F16 -> Q8Exact -> F32 | `0.00287484` |
| F32 -> Q8Exact -> F32 | `1.90735e-06` |

F32 input and output therefore produce the closest isolated V result.
However, replaying each variant through GQA, the established FP16 attention
boundary, and the real GPU O/FFN tail does not close the full layer: even the
best local V policy leaves approximately `0.204956` layer-output error. A
`0.000244141` attention difference after materialization is sufficient to
alter the downstream quantized O projection materially. V local parity and
end-to-end causal parity are distinct contracts.

The external-attention injection remains the downstream control and produces
`0.00282186` GPU layer-output error. B18 therefore does not justify a
production precision change; it narrows the next investigation to the
V-to-attention materialization/quantization boundary. No tolerance changed.

**M4-B18 status: COMPLETE DIAGNOSTIC SLICE; M4-B remains OPEN.**
