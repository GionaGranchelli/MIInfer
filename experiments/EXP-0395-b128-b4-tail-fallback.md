# EXP-0395 — Legacy B4 tail fallback for logical B128

**Status:** `REVERTED`
**Disposition:** `REJECT`
**Date:** 2026-09-24
**Base commit:** `ecbdb71`

## Question

Can the existing narrow/B4 layer-major architecture replace only the
pathological logical B128 tail after a qualified B512 prefix, without
reopening the rejected whole-model B4 architecture?

Whole-model B4 remains rejected by EXP-0285. This experiment tested only the
new hybrid `B512 + legacy-B4 B128 tail` premise.

## Candidate

The temporary candidate was opt-in as `MIINFER_EXP0395_B128_B4_TAIL=1`. For a
tail at base 512 with count 128 it bypassed the B128-wide route and forced the
existing narrow layer-major machinery: `prepare_prefill_batch`, per-token
causal ordering, and `finish_prefill_batch`/`finish_prefill_batch4`.

B512 remained on the existing full-layer-major route. No kernel, attention
algorithm, precision, or B128 scheduler redesign was added.

The candidate was reverted after the required execution gate failed.

## Route accounting

The candidate added host-only counters for:

```text
b4_tail_tokens
b4_tail_groups
b4_recurrent_layers
b4_attention_layers
```

The intended exact P640 candidate contract was `128` tail tokens and `32`
B4 groups, with recurrent and attention layer counts separated. The run did
not complete a result line before termination, so no completed runtime counter
record can honestly be claimed. P896 and P1022 were not attempted.

## Correctness and performance gate

The valid matrix used the exact 640-token prompt, current B128-wide control
(`MIINFER_EXP0374_REMAINDER_SCHED=1`), same engine, `reuse_session=false`, and
counterbalanced C/A ordering with 16 generated tokens requested.

The candidate matrix produced no completed pair result within a bounded
multi-minute observation window. The sole ordinary P640 control sanity run
completed at:

```text
processed tokens: 640
prefill: 16283.02 ms
total:   16286.29 ms
```

The candidate remained actively consuming GPU/CPU resources for more than
four minutes without returning a correctness result. This is an immediate
performance/completion failure against a roughly 16-second control and cannot
pass the required correctness-first gate. No continuation, final-hidden,
logit, or top-10 comparison is reported because the candidate never completed.

The first exploratory invocation without the required B128-wide control
selector was discarded from evidence: it selected the existing narrow path on
both sides and was not the intended A/B.

## Decision

`REVERT`. The legacy-B4 B128-tail fallback did not demonstrate a completed
semantic result and was already far outside the allowed performance envelope.
No B4 stage profiling or optimization follow-up is authorized.

- B128-wide status: **PATHOLOGICAL**.
- B4 whole-model architecture: **REJECTED**.
- B4 tail fallback: **REJECTED**.
- B64: **REJECTED**.
- B4: **BLOCKED** outside this rejected tail experiment.

**ONE next PRIMARY:** choose a materially different bounded B128 intervention;
do not reopen legacy-B4 profiling or optimization.

The legacy-B4 B128-tail fallback was reverted. No B4 profiling or optimization
follow-up is authorized.

## Provenance

- experiment commit: `a261d70`
- graph refresh commit: `1aa447c`
- graph SHA-256: `fdbc5a8343a79ba49ea67a25ce89789ebecdd45fccc26421f0c3d9a136972e48`
- final source tree contains no EXP-0395 candidate or test harness.
