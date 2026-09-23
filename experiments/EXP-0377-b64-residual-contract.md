# EXP-0377 — Qualify Existing B64 Residual Contract

## Question

Can the existing 64-token prefill contract replace the first 64 tokens of a
sub-128 residual, leaving only the `<64` fallback unchanged?

## EXP-0376 authorization

EXP-0376 measured 63,725.84 ms of residual-associated P1022 wall time
(59.64%). Repeated B128 was qualified and B64/B4 residual work was authorized.
This experiment tests exactly one variable: opt-in use of the existing B64
contract. B4 scheduling is explicitly excluded.

## Protocol gate

```text
Objective: reduce the scalar-covered residual from 126 to 62 tokens.
Current bottleneck: P1022 residual-associated wall, 63,725.84 ms.
Reference: EXP-0376 qualified B512/B128 route and scalar residual fallback.
Difference: schedule R>=64 as existing count=64 batch work.
Expected impact: remove 64 whole-model scalar tokens; no kernel speed claim.
Independent variable: EXP0377 B64 residual selector.
Resource gate: no new allocation, workspace, kernel, or representation.
Correctness gate: model-boundary hidden/logits/tokens/state checks.
Performance gate: clean P960/P1022 timing and B64 HIP events.
Kill criteria: fallback/unsupported layer, material divergence, or no benefit.
Success: semantically valid existing B64 route and material P1022 recovery.
```

## Existing B64 contract audit

The current `kPrefillBatch=64` APIs already accept count 64:

- recurrent preparation uses existing normalized/QKV preparation in groups of
  four; `finish_prefill_batch()` uses the existing 64-token GDN chunk path and
  existing deferred FFN tail path, but the caller still invokes scalar
  `layer.run` once per token because the non-scalar wide-core guard requires
  count >=128;
- full-attention preparation accepts count 64; count 64 uses the existing
  prepared projection path, causal/KV path, and existing deferred tail;
- attention deferred output uses `finish_prefill_batch()`, which selects the
  existing wide batch path when its existing MMQ resources are present, or
  the existing B4 tail loop otherwise;
- final output is the existing deferred batch output and residual add.

No new Q/K/V, GDN, GEMM/MMQ, attention, quantization, or padding kernel is
introduced. The scheduler only changes the first residual chunk size.

## Scheduler decomposition

With the opt-in selector:

```text
R < 128 and R >= 64 -> count=64, then R -= 64
R < 64              -> existing scalar fallback
```

Expected P1022 decomposition is B512 + 3×B128 + B64 + 62 scalar tokens.

## Results

The first exact zero-residual test was P960:

```text
B512 + 3xB128 + B64 = 960 tokens
prefill wall: 75638.34 ms
```

The scheduler trace correctly emitted:

```text
EXP0377 route base=896 count=64 path=existing_b64_prefill
```

However, the existing layer-major implementation did not provide a whole-
model B64 execution contract. The counters reported:

```text
complete_b512=1
partial_b128=3
partial_b64=1
residual_tokens=0
scalar_layer_run_calls=64
scalar_layer_run_tokens=4096
```

That is one scalar `layer.run` call per layer for all 64 B64 tokens. The
deferred tail machinery is batch-capable, but the recurrent and attention
preparation/core path still enters the scalar per-token layer route because
the existing wide path is selected only for counts >=128. Therefore this is
not a valid B64 replacement for whole-model scalar residual work.

The semantic gate was stopped immediately as required. No continuation,
hidden/logit comparison, or P1022 performance claim was made for this mixed
route. The B64 premise failed before those gates.

The B64-specific HIP event was intentionally not collected: EXP-0376 timing
events cover complete B512/B128 chunks, and the required stop rule prohibits
continuing measurement after the unexpected fallback invalidated the B64
premise. Consequently B64 GPU/host cost, P1022-after-B64, and regression
controls are not applicable to this rejected candidate.

## Decision

**REJECT.** The opt-in scheduler split is structurally correct, but the
existing count-64 path falls back to scalar layer execution for every layer.
It does not satisfy the experiment's B64 contract and must not be retained as
a performance candidate.

## Exact next PRIMARY

Identify whether an already-existing non-scalar recurrent/attention count-64
core contract exists independently of the scheduler. If not, record the
missing primitive and stop; implementing that primitive is outside EXP-0377.

B4 scheduling is not authorized.

No new GPU math kernel was implemented.
