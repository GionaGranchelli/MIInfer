# EXP-0343 — M25 faithful GDN state input/output contract

**Status:** REJECT
**Milestone:** M25 stretch follow-up
**Date:** 2026-09-12
**Baseline:** `a1b4524`
**Reference:** `mx-llama.cpp@2e9d29fe736969160f17476ec6f0a6298cee6966`

## Hypothesis

The pinned oracle's GDN kernel receives the previous recurrent state through a
read-only pointer and writes the updated state through a separate pointer. MIInfer
used one in/out pointer for its equivalent scan. Giving the compiler the same
non-aliasing state contract might recover the oracle's standalone speed without
changing the mathematical state layout.

## Candidate

The Mx GDN kernel was changed to accept `const state_input` and `state_output`
with `__restrict__` qualifiers. The standalone B512 harness used separate device
buffers. The production prefill path allocated a second 3 MiB state buffer per
recurrent layer and copied the canonical state into it before the scan; the scan
then wrote the canonical state output.

This was a contract-level test of the pinned
`gated_delta_net_chunked_cuda` behavior, not a launch-geometry-only transplant.

## Environment

- AMD Instinct MI50 / gfx906 / Wave64
- clean `env -i` process
- standalone deterministic fixture, 512 tokens, 16 key heads, 48 value heads,
  state size 128, Mx Tc2 mapping
- no profiler; this was a primitive screen, not a P512 qualification

## Correctness

The separate-pointer candidate completed without non-finite values. Against the
existing recurrent fixture it reported:

```text
Mx max output error: 1.5e-8
Mx max state error:  8.9e-8
```

These match the baseline's tight recurrent errors.

## Measurements

Three clean standalone samples were collected after rebuilding each variant:

```text
control:  2822.075, 2829.595, 2836.316 us  (median 2829.595 us)
split:    2827.675, 2838.396, 2849.435 us  (median 2838.396 us)
```

The candidate was `+8.801 us`, or `+0.311%`, slower. This is before charging
the production path for the state copy. The extra representation would add
3 MiB per recurrent layer, about 144 MiB across 48 layers, and one 3 MiB
device-to-device copy per recurrent layer for each P512 prefill.

## Decision

**REJECT.** The faithful separate-state contract is numerically safe but does
not improve the MIInfer GDN primitive. Its additional persistent memory and
copy make it unsuitable for production or an end-to-end promotion experiment.
The working-tree candidate was removed; the one-pointer Mx GDN path remains
the accepted default.

## Follow-up

Do not retry state-pointer splitting alone. Continue the M25-L differential at
the full QKV/GDN execution boundary, including materialization and composition,
only if a measured end-to-end gap remains after fresh oracle requalification.
