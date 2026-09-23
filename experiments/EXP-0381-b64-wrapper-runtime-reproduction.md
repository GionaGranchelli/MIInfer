# EXP-0381 — Isolate the Historical B64 Stop Above the Proven Prefill Boundary

## Question

Can the historical EXP-0379 Candidate-B stop be reproduced after the exact
P960 B64 prefill has completed, and if so, which ordinary wrapper operation
fails to return?

## EXP-0379 historical stop

Candidate B used the repeated-B128 scheduler plus the opt-in B64 composition
route on `P960 = B512 + 3xB128 + B64`. It entered B64 but remained in `D`
state for approximately 5:35 with GPU utilization at 0%. The candidate was
reverted. EXP-0380 later proved the exact B64 chunk through terminal prefill
completion with bounded exits.

## EXP-0380 proven boundary

Current HEAD proves the exact P960 B64 route through:

```text
B512 + 3xB128 + B64
all recurrent layers
all 16 attention layers and stages
layer handoffs
B64 chunk completion
prefill_layer_major return
on_prefill_state / on_prefill_complete
terminal stream synchronization
```

EXP-0381 used that same route, without an EXP-0380 forced exit, and added only
test-only wrapper markers. Every admitted run reached `MODEL_INIT_END`,
`PROMPT_BEGIN`, `Prompt tokens: 960`, and `B512_BEGIN`.

## Wrapper configuration

The admitted current-HEAD runs used:

```text
MIINFER_PRESET=m25_hi_qualified
MIINFER_EXP0374_REMAINDER_SCHED=1
MIINFER_EXP0377_B64_RESIDUAL=1
MIINFER_EXP0381_WRAPPER_PROBE=1
context=1024
prompt=960 tokens (The repeated 959 times)
```

The historical EXP-0379 candidate selector was reverted and is not present in
the current executable. Its effective current equivalent is the two selectors
above that enable the already-proven repeated-B128/B64 test route. No session
reuse was requested; no graph decode was selected for the direct two-token
case (`stream=true`).

The exact old shell command was checked against the repository records and
available shell history. The reverted `MIINFER_EXP0379_B64_CORE=1` selector is
not recoverable as an executable current-HEAD invocation; the current-equivalent
wrapper command above is therefore the reproducible admitted command.

## Zero-token wrapper result

Case A used `max_new_tokens=0` and the ordinary CLI wrapper. It emitted:

```text
MODEL_INIT_END
PROMPT_BEGIN
B512_BEGIN
PREFILL_RETURNED
PREFILL_CALLBACKS_RETURNED
ZERO_TOKEN_RETURN_BEGIN
ZERO_TOKEN_RETURN_END
GENERATE_RETURN
CLI_WRAPPER_RETURN
```

Result: **PASS**. The exact B64 prefill unwinds through the ordinary wrapper.

## First-token ladder

Case B used `max_new_tokens=1`. Every stage returned and synchronized:

```text
FIRST_TOKEN_BEGIN
FINAL_NORM_BEGIN / FINAL_NORM_RETURN / FINAL_NORM_GPU_COMPLETE
FINAL_QUANT_BEGIN / FINAL_QUANT_RETURN / FINAL_QUANT_GPU_COMPLETE
LM_HEAD_BEGIN / LM_HEAD_RETURN / LM_HEAD_GPU_COMPLETE
ARGMAX_BEGIN / ARGMAX_RETURN / ARGMAX_GPU_COMPLETE
TOKEN_COPY_READ_BEGIN / TOKEN_COPY_READ_END
FIRST_TOKEN_END
GENERATE_RETURN
CLI_WRAPPER_RETURN
```

Semantic sanity:

```text
first token id = 561
final norm finite = 1
logits finite = 1
```

Result: **PASS**. No post-prefill stage failed to return or synchronize.

## One-step direct decode

Case C used `max_new_tokens=2` and `stream=true`, forcing the ordinary direct
`step()` path rather than HIP graph decode. It emitted:

```text
FIRST_TOKEN_END
ON_FIRST_TOKEN_BEGIN / ON_FIRST_TOKEN_END
STEP_BEGIN / STEP_RETURN
TOKEN_CALLBACK_BEGIN / TOKEN_CALLBACK_END
GENERATE_RETURN
CLI_WRAPPER_RETURN
```

Result: **PASS**. The exact B64 prefill state transitioned through one direct
decode step and the streaming callback.

## Initialization contamination

One early attempt was intentionally excluded: it inherited
`MIINFER_EXP0380_B64_ATTN_PROBE=1` without a nonzero stage and exited at the
EXP-0380 stage-0 preparation checkpoint. It is not evidence about EXP-0381.

The admitted A/B/C runs all reached model initialization, prompt, and B512
markers. No admitted run stalled before those markers. No admitted run
entered D state. No new ROCm page fault or thermal/clock anomaly was observed.

## D-state evidence

No admitted EXP-0381 case reproduced the historical D-state stop, so there is
no new `wchan` or stack classification. The prior EXP-0379 classification
remains historical `UNKNOWN / runtime composition wait`; current A/B/C results
do not support a GPU noncompletion claim.

## Decision

**QUALIFY_RUNTIME.** The ordinary current-HEAD wrapper passes zero-token
return, first-token production, and one direct decode transition after the
exact P960 B64 prefill. The historical EXP-0379 Candidate-B stop is not
reproducible in the current wrapper path and no first missing END marker exists.

## Exact next PRIMARY

Reintroduce the B64 composition as the next opt-in semantic qualification
experiment and run the previously blocked P960 semantic checks, followed by
the P1022 attribution/performance gate. Do not investigate B64 kernels again.
B4 remains blocked until that qualification succeeds.

## Required disposition

```text
Historical EXP-0379 hang reproducible: NO
B64 scheduler qualification authorized: YES, opt-in next experiment only
B4 authorized: NO
```

No performance benchmark or new GPU math kernel was implemented.

## Provenance

Experiment SHA: `6b1b699cc5030798b7fec9caee1bc6c4cffecbb0`.

Graph SHA: `4102b5a9492f9c709969c36a8d039ff5c8b79dad` (`graphify-out/graph.json` blob).

Working-tree status: clean after the experiment record and graph refresh.
