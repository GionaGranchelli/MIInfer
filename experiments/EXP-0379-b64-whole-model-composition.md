# EXP-0379 — B64 whole-model composition

## Question

Can the already-proven B64 recurrent and attention primitives be composed into
the complete model route at `P960 = B512 + 3×B128 + B64`, without adding a
kernel or changing the scheduler contract?

## EXP-0375 basis

EXP-0375 qualified the corrected repeated-B128 scheduler, including later-base
B128 execution. EXP-0378 established that the underlying B64 primitives exist;
the remaining blockers were the `>=128` composition guards. EXP-0376 measured
the P1022 residual-associated wall at 63,725.84 ms (59.64%).

## Candidate

The candidate was opt-in under `MIINFER_EXP0379_B64_CORE=1` and required
`MIINFER_EXP0374_REMAINDER_SCHED=1`. It minimally relaxed the recurrent wide
entry guards and the attention batched-resource/selector guards for exactly 64
tokens. No GPU kernel, layout, geometry, B512, B128, decode, or B4 code was
changed.

The candidate was reverted after the runtime-safety gate failed. The default
runtime therefore remains unchanged.

## Clean current-HEAD timing

Not qualified. The experiment stopped at the first invalid candidate, before
the requested P512/P640/P768/P896/P1024 matched timing matrix. Existing EXP-0376
reference data in the same build was retained for the already-qualified
chunks:

| Chunk | Base | Count | GPU ms | Host wall ms | Host-GPU gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| B512 | 0 | 512 | 2483.73 | 35.2465 | -2448.48 |
| B128 #1 | 512 | 128 | 14732.6 | 14231.1 | -501.491 |
| B128 #2 | 640 | 128 | 14602.9 | 16463.6 | 1860.77 |
| B128 #3 | 768 | 128 | 13398.4 | 13288.5 | -109.911 |
| B64 | 896 | 64 | not reached | not returned | candidate hung |

The negative gaps reflect the existing event/host timing scopes and are not
interpreted as physical negative overhead.

## Route and runtime-safety results

The first P960 structural run, with recurrent guards relaxed but before the
attention resource guard was relaxed, reached the B64 chunk and reported:

```text
EXP-0374 remainder scheduler: complete_b512=1 partial_b128=3 partial_b64=1 residual_tokens=0
scalar_layer_run_calls=16 scalar_layer_run_tokens=1024
```

This failed the hard composition gate: the 16 attention layers still used
scalar `layer.run` calls and attention was not batched.

After the attention resource guard was also relaxed, the same-build P960
candidate entered the B64 execution but did not return. The process remained
in `D` state for about 5:35 with GPU utilization at 0%; it was killed. No
semantic, boundary, P1022, or allocation conclusion was drawn from this run.

The failure is therefore not attributable to causal-context scaling. The
whole-model B64 route was not qualified, and no B64 GPU-event timing exists.

## Single-B128 restored-state matrix

Not run. The candidate failed before B64 qualification, and EXP-0379 requires
stopping immediately on an invalid candidate rather than broadening the test.

## Expected causal-attention scaling

For a valid B64 comparison, causal work should grow with absolute base
position. EXP-0379 produced no valid B64 measurement with which to compare that
structural increase against invocation order.

## Execution-mode counters

The only completed B64 counter sample contained 16 scalar layer calls and
1024 scalar tokens. This is a rejected route proof, not a qualified internal
plan. The subsequent candidate hang prevented a complete counter sample.

## Coarse stage attribution

Not applicable. GPU timing could not be collected for the candidate B64 chunk,
so no stage was profiled.

## Allocation/setup audit

The completed B512/B128 chunks showed zero allocation deltas in the existing
EXP-0376 instrumentation. The hung B64 candidate did not reach a terminal
event/counter record. No claim about B64 setup repetition is made.

## Hardware telemetry

During the hung candidate: SCLK 1606 MHz, MCLK 1000 MHz, 225 W policy,
temperature approximately 34–36 °C, GPU utilization 0%. Invalid fan-RPM
telemetry was ignored. The evidence is inconsistent with a thermal or clock
throttle explanation.

## P1022 corrected residual attribution

Not recalculated. Because whole-model B64 failed its hard gate, EXP-0379 did
not use the candidate to decompose P1022. The last qualified EXP-0376 result
remains:

```text
B512 + B128 #1 + B128 #2 + B128 #3 = 43123.84 ms
P1022 total = 106849.68 ms
remaining residual-associated wall = 63725.84 ms (59.64%)
```

## Decision

**REJECT** the EXP-0379 whole-model B64 composition candidate. The primitive
contract from EXP-0378 is insufficient evidence for safe whole-model
composition: the first guard relaxation left scalar attention calls, and the
complete guard relaxation hung with the GPU idle. No code from the candidate is
retained.

## Exact next PRIMARY

Isolate the first B64 recurrent/attention operation that hangs in the
whole-model composition using a bounded, layer-scoped runtime probe or existing
fixture machinery. The next work is runtime-contract isolation, not B4,
residual scheduling, or a new GPU math kernel.

## Provenance

Experiment SHA: `85299fe14092a44b3b4dc41a17c1220defa74ad1` (record commit; provenance amendment follows).

Graph SHA: `43a31e5027fa721a43450f680c5320cba24197d7` (`graphify-out/graph.json` blob).

Working-tree status: clean after the experiment record and graph refresh.

No new GPU math kernel or residual scheduler optimization was implemented.
