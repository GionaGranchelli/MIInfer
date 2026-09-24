# EXP-0382 — Revalidate the Actual Composed-B64 Runtime

## Question

Does the actual composed `B512 + 3xB128 + B64` route preserve the semantic
contract needed to authorize B64 timing and the later B4 question?

## EXP-0375 basis

EXP-0375 corrected the repeated-B128 dispatch contract and qualified the
zero-residual B128 routes. EXP-0381 established wrapper markers but, as
corrected below, did not enable the actual count-64 composition guards.

## EXP-0381 scope correction

EXP-0381's admitted A/B/C runs used the wrapper probe and repeated-B128/B64
scheduler selectors, but did not enable the count-64 composition guards. They
proved wrapper lifecycle only. The run that inherited the EXP-0380 selector
was explicitly excluded. EXP-0381 is therefore classified:

```text
WRAPPER_LIFECYCLE_PASS
B64_COMPOSED_RUNTIME_NOT_PROVEN_BY_EXP0381
```

EXP-0380 remains the terminal-prefill authority.

## New selector

Added the opt-in test-only selector:

```text
MIINFER_EXP0382_B64_COMPOSE=1
```

It requires the EXP-0374 scheduler, uses the EXP-0377 B64 residual route,
and enables only the already-proven count-64 recurrent and batched-attention
composition guards. It contains no EXP-0380 stage exits or diagnostic exits.
No production default, scheduler redesign, or GPU kernel changed.

## P960 route proof

Exact prompt: 960 tokens, `P960 = B512 + 3xB128 + B64`.

The clean current-HEAD route proof emitted:

```text
complete_b512=1
partial_b128=3
partial_b64=1
residual_tokens=0
scalar_layer_run_calls=0
scalar_layer_run_tokens=0
```

The count-64 route therefore used the composed wide recurrent and batched
attention path rather than scalar fallback.

## Composed-B64 wrapper ladder

### A — zero generated tokens

**PASS.** `max_new_tokens=0` emitted model-init, prompt, B512, prefill return,
callback return, zero-token return, `generate` return, and CLI wrapper return.

### B — first token

**PASS for lifecycle, but semantic result diverges.** Final norm, final
quantization, LM head, argmax, token read, and wrapper return all completed.

```text
first token id: 248046
final norm finite: 1
logits finite: 1
```

The scalar-residual EXP-0381 control produced first token ID `561` for the
same prompt and wrapper conditions.

### C — one direct decode step

**FAIL / not reached.** With `max_new_tokens=2` and `stream=true`, the
composed route produced first token `248046`; the ordinary generation loop
classified it as a stop token and returned without entering `step()`. No
`STEP_BEGIN` marker was emitted. This is not a host hang: it is an immediate
composed-route token/termination divergence.

Per the stop rule, semantic qualification and timing were not run.

## P960 semantics

Not run. Case C failed before the direct decode transition, and the first
token already differs from the qualified scalar-residual control. No broad
tensor archaeology was started.

## P960/P1022 timing

Not run. The candidate failed the composed runtime/semantic gate before
performance qualification. Consequently the clean current-HEAD timing set is:

```text
P512/P640/P768/P896/P1024: not run in EXP-0382
```

The experiment intentionally stopped before collecting performance data.

## Required per-B128 timing table

Not collected because the A/B/C stop rule fired before performance
qualification. The EXP-0375 repeated-B128 timing remains prior evidence, not
an EXP-0382 measurement.

| Chunk | Base | Count | GPU ms | Host wall ms | Host/GPU gap | Increment vs first B128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B512 | 0 | 512 | not run | not run | not run | not run |
| B128 #1 | 512 | 128 | not run | not run | not run | baseline unavailable |
| B128 #2 | 640 | 128 | not run | not run | not run | not run |
| B128 #3 | 768 | 128 | not run | not run | not run | not run |

## Single-B128 restored-state matrix

Not run. No state checkpoint or timing claim is made by this experiment.

## Expected causal-attention scaling

Not measured. The expected increase with causal prefix length is not used to
explain the semantic failure.

## Execution-mode counters

The P960 route proof reported three B128 chunks and one composed B64 chunk,
with zero scalar calls. No later stage counter comparison was run after the
wrapper ladder failed.

## Allocation/setup audit

Not run. No allocation or repeated-setup conclusion is claimed.

## Hardware telemetry

No performance run was admitted. The existing qualified environment remains
MI50/gfx906 at `SCLK=1606 MHz`, `MCLK=1000 MHz`, `225 W` policy, with the
external physical fan at full speed; invalid fan-RPM telemetry is ignored.

## Allocation and regression gates

Not run. No B64 timing or allocation claim is made. P512/P768/P1024 controls
were not broadened after the candidate failed the required ladder.

## Decision

**REPRODUCED_FAILURE.** The clean non-diagnostic composed-B64 route has zero
scalar work and passes zero-token plus first-token execution, but its first
token differs from control and terminates the ordinary two-token direct-decode
case before `step()`. The exact next boundary is the first composed-B64
semantic divergence at final-hidden/logit/token selection, not a B64 kernel
hang.

## Exact next PRIMARY

Localize only the first composed-B64 model-boundary semantic divergence needed
to explain token `248046` versus control token `561`, starting with final
hidden/logit comparison at P960. Do not run P1022 timing, optimize residuals,
or authorize B4 until the composed route passes semantic qualification.

```text
B64 scheduler qualification: NOT QUALIFIED
B4 authorized: NO
```

No new GPU math kernel was implemented.

## Provenance

Experiment SHA: `acb7ee6808b19d698f48b952be2ea7a75b72df7c`.

Graph SHA: `f9578e8d17293dfb4797a31e7a2d8dab28686fdc`
(`graphify-out/graph.json` blob).

Working-tree status: clean after the experiment record and graph refresh.
