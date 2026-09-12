# EXP-0344 — M25 oracle GDN launch-bounds contract

**Status:** REJECT
**Milestone:** M25 stretch follow-up
**Date:** 2026-09-12
**Baseline:** `3de0a0d`
**Reference:** `mx-llama.cpp@2e9d29fe736969160f17476ec6f0a6298cee6966`

## Hypothesis

The pinned oracle declares its 128-thread, `64x2` GDN block with
`__launch_bounds__(256, 2)`, while MIInfer declares the same actual launch with
`__launch_bounds__(128, 2)`. Matching the oracle's compiler/resource contract
could change register allocation and recover its measured GDN latency.

## Candidate

Only the Mx GDN kernel declaration changed from `__launch_bounds__(128, 2)` to
`__launch_bounds__(256, 2)`. The grid, block, state layout, arithmetic, and
all other runtime behavior were unchanged.

## Environment

- AMD Instinct MI50 / gfx906 / Wave64
- clean `env -i` process
- standalone deterministic fixture, 512 tokens, 16 key heads, 48 value heads,
  state size 128, Mx Tc2 mapping
- no profiler; this was a primitive screen, not a P512 qualification

## Correctness

The candidate completed without non-finite values and retained the existing
Mx fixture errors:

```text
Mx max output error: 1.5e-8
Mx max state error:  8.9e-8
```

## Measurements

The prior three-sample control median was `2829.595 us`:

```text
control:   2822.075, 2829.595, 2836.316 us  (median 2829.595 us)
candidate: 2932.156, 3001.277, 2868.958 us  (median 2932.156 us)
```

The oracle launch-bounds declaration was `+102.561 us`, or `+3.625%`, slower
on this MI50. The actual production launch remains 128 threads, so this result
does not imply a change to the accepted kernel geometry.

## Decision

**REJECT.** The pinned launch-bounds resource contract is slower for MIInfer's
adapted GDN kernel. The candidate was removed; MIInfer retains
`__launch_bounds__(128, 2)`.

## Follow-up

Do not retry this declaration alone. Remaining stretch work must measure a
different execution/dataflow contract or a primitive with a positive exact-shape
differential.
