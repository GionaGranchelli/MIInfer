# EXP-0365 — Context scaling reconciliation

## Question

Why does the current MIInfer Qwen3.8-27B-Q4_K_M prefill path jump from about
2.5 s at P512 to tens of seconds around P2K while mx remains approximately
linear?

This is measurement and diagnosis only. No production code, optimization, or
math kernel was changed. EXP-0360 through EXP-0363 were not reopened.

## Environment

| Item | Value |
| --- | --- |
| MIInfer commit | `cfc44e4` graph-refresh tree; runtime source at `d20ada5` |
| Model | Qwen3.8-27B-Q4_K_M; existing hash `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169` |
| GPU | AMD Instinct MI50/MI60, gfx906, Wave64 |
| SCLK/MCLK | fixed 1606/1000 MHz |
| Power cap | 225 W |
| External fan | physically fixed at full speed; ROCm fan telemetry ignored |
| Route | explicit expansion of the `m25_hi_qualified` selector vector for capacity-only runs |
| Prompt | exact 512-token repeated-fox prompt unless otherwise stated |

The capacity-only matrix retained 612 telemetry samples. SCLK and MCLK were
1606/1000 MHz in every sample; junction temperature was 34–59 C, package
power 28–196 W, and the reported cap was 225 W. The capacity-only results are
therefore hardware-state qualified. The later boundary probes did not retain a
separate telemetry file and are diagnostic corroboration, not new qualification
points.

## EXP-0364 anomaly

EXP-0364 measured approximately 2.554 s at P512, 86.754 s at 2042 tokens,
94.521 s at 4082 tokens, and 120.282 s at 8162 tokens. mx measured 2.300 s,
9.258 s, 18.706 s, and 38.225 s at the corresponding nominal points.

## Capacity-only matrix

The prompt remained exactly 512 tokens while only
`MIINFER_CONTEXT_CAPACITY` changed. The same explicit selector vector was used
for every run.

| Capacity | Prompt tokens | Prefill ms | Tok/s | Allocated bytes | Peak bytes |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 512 | 2493.98 | 205.29 | 21,993,243,028 | 21,993,243,028 |
| 2048 | 512 | 2451.11 | 208.89 | 22,060,355,988 | 22,060,355,988 |
| 4096 | 512 | 2454.45 | 208.60 | 22,194,581,908 | 22,194,581,908 |
| 8192 | 512 | 2492.25 | 205.44 | 22,463,033,748 | 22,463,033,748 |

The ~469.8 MB allocation increase is consistent with capacity-sized KV/state
storage. It does not produce a tens-of-seconds P512 slowdown.

**H1 rejected:** context capacity alone is not the discontinuity.

## Prompt-length boundary matrix

With capacity fixed at 8192 and the same explicit selector vector:

| Prompt tokens | Prefill ms | Tok/s | Interpretation |
| ---: | ---: | ---: | --- |
| 512 | 2492.25 | 205.44 | one complete B512 fast chunk |
| 513 | 3032.70 | 169.16 | one B512 fast chunk plus one slow tail token |
| 1022 | 80973.89 | 12.62 | one B512 fast chunk plus 510 slow tail tokens |
| 1023 | 81510.50 | 12.55 | one B512 fast chunk plus 511 slow tail tokens |
| 2042 | 86753.70 | 23.54 | three B512 fast chunks plus a 506-token slow tail |
| 4082 | 94521.30 | 43.19 | seven B512 fast chunks plus a 510-token slow tail |
| 8162 | 120281.76 | 67.86 | fifteen B512 fast chunks plus a 482-token slow tail |

The threshold is therefore not 1024 tokens. It begins at any non-empty
remainder after a complete B512 chunk; the large visible discontinuity appears
when that remainder approaches a full 512-token chunk.

## Preset versus explicit-vector equivalence

The canonical `MIINFER_PRESET=m25_hi_qualified` dump sets the same effective
selectors used by the explicit vector and fixes capacity to 1024. The
capacity-only explicit 1024 run and the earlier fresh preset P512 run both
reported:

```text
device_allocation_count = 2685
device_allocated_bytes  = 21993243028
prefill chunk            = 512
wide/layer-major/full-layer-major = enabled
```

The explicit run measured 2493.98 ms; the earlier preset run measured 2553.84
ms. This one-run difference is not a qualified timing conclusion and is within
the spread of the existing six-pair control (EXP-0354 median 2486.685 ms).
No effective selector or allocation mismatch was found. The long-context
curve is therefore not explained by a demonstrated preset/vector mismatch.

## Execution-contract fingerprint

The relevant fingerprint is stable across the diagnostic runs:

```text
MIINFER_CONTEXT_CAPACITY       variable as tested
MIINFER_PREFILL_CHUNK          512
MIINFER_PREFILL_LAYER_MAJOR   1
MIINFER_PREFILL_WIDE_CHUNK    1
MIINFER_PREFILL_FULL_LAYER_MAJOR 1
MIINFER_PREFILL_WIDE_ATTN     1
MIINFER_MX_GDN                1
MIINFER_*_MMQ selectors       same qualified vector
workspace batch capacity      B512 for full-layer-major prefill
```

No alternate mathematical kernel route was selected by increasing capacity.
The route transition is caused by the prompt remainder and the existing
partial-chunk fallback.

## Capacity-dependent source audit

Relevant uses were classified as follows:

| Use | Classification | Finding |
| --- | --- | --- |
| `key_cache`/`value_cache` allocation and memset | allocation/copy size | scales with `g_cache_capacity`; explains ~470 MB, not 80 s |
| KV strides and attention bounds | layout/loop bound | use capacity as stride/bound; no P512 capacity-only timing effect |
| `d_decode_tokens_` allocation | allocation size | scales with capacity; small relative to the model |
| shared wide-prefill workspace | workspace size | configured by `MIINFER_PREFILL_CHUNK`, remains B512 |
| `prefill_a_`/`prefill_b_` in full-layer-major mode | workspace size | explicitly allocated at `kFullPrefillCapacity` = 512 |
| `prefill_layer_major` outer loop | prompt/chunk loop | processes prompt in requested chunks, then remainder |
| `prefill_full_layer_major_chunk` selection | route condition | only `count <= 512 && count % 512 == 0` |
| `prepare_prefill_batch` | fallback condition | accepts only `count == 512` or `count == prefill_capacity` |
| partial remainder | execution route | falls through to per-token `layer.run` |

## Work and dispatch census

For a prompt of `512 + r` tokens with `0 < r < 512`, the current code does:

```text
one complete B512 full-layer-major pass
+ r sequential token passes through all 64 layers
```

At 1022 tokens, `r = 510`, so the partial path executes approximately
`510 × 64 = 32,640` layer-level runs after the first fast B512 pass. The
normal B512 profile counted 48 recurrent and 16 attention family invocations
for one wide pass. The partial path invokes those per-token layer operations
repeatedly rather than using the B512 wide batch contract. This is the
discontinuous work multiplier; it is not an attention geometry hypothesis.

The pre-existing diagnostic counters are launcher-family counters rather than
a total HIP dispatch census. No new intrusive per-kernel synchronization was
added. The source-level work count is sufficient to explain the measured
multiplier; exact HIP total counts were not required to identify the branch.

## Wall-clock reconciliation

The clean capacity-only matrix provides the normal-point wall clock and the
boundary matrix provides the pathological-point wall clock. Source control
flow reconciles the difference:

| Point | Complete B512 work | Partial fallback work | Wall time |
| --- | --- | --- | ---: |
| P512 | 1 fast full-layer-major pass | none | 2.49 s |
| P513 | 1 fast pass | 1 sequential layer pass | 3.03 s |
| P1022 | 1 fast pass | 510 sequential layer passes | 80.97 s |
| P2042 | 3 fast passes | 506 sequential layer passes | 86.75 s |

This accounts for the observed shape without attributing it to a math-kernel
slowdown. Initialization/allocation is reported separately in every log and
does not change from the P512 capacity matrix into the pathological prompt
case. A 95% event-level exclusive timing reconciliation was not necessary to
identify this source-visible repeated-work cause; adding per-kernel
synchronization would have contaminated the diagnosis. The remaining
unattributed portion is ordinary launch/kernel variation around the measured
wall time.

## Hardware qualification

The capacity-only matrix had stable clocks and no observed thermal/power
transition capable of explaining the jump. The 8192-capacity P512 run was
205.44 tok/s, while the 1022-token fixed-capacity run was 12.62 tok/s under the
same selector vector. This rules out a capacity-triggered hardware state
change for the discontinuity.

## Root cause

**Confirmed:** the full-layer-major prefill implementation only accepts complete
B512 chunks. Any remainder after a complete chunk falls through to the
per-token `layer.run` path. Near 1024 tokens this means roughly 510 sequential
passes through all 64 layers, producing the ~81 s latency. At larger prompts,
the same slow tail is added after each set of complete B512 chunks, explaining
the EXP-0364 curve.

This is a route/control-flow defect, not evidence that attention, FFN, GDN, or
dispatch overhead should be optimized independently.

## Amdahl ceiling

At P1022, removing the 510-token sequential-tail work would reduce the
measured 80.97 s toward the one-B512 baseline plus the cost of a correct tail.
The observed removable excess over P512 is approximately:

```text
80.974 s - 2.492 s = 78.482 s
78.482 / 80.974 = 96.9% of pathological wall time
```

This is a diagnosis ceiling, not authorization to implement the fix in this
experiment.

## Decision

**LEARN.** H1 is rejected. H2 is confirmed at the B512 chunk boundary. H3 is
not supported by selector/allocation evidence. H4 is rejected for the
capacity-only matrix. H5 is unnecessary: the repeated partial-tail route
accounts for the discontinuity.

## Next PRIMARY frontier

The next primary objective is measurement/contract work only:

> Define and qualify the correct partial-tail execution contract for a prompt
> remainder after B512 without changing production behavior in this goal.

Before implementation, the next experiment must specify whether a smaller
valid batched path, a deliberately bounded tail route, or another existing
contract preserves recurrent state, causal KV semantics, and numerical
correctness. It must include a clean-vs-diagnostic timing plan and an explicit
gate preventing reintroduction of the per-token fallback. No new math kernel
or optimization was implemented here.
