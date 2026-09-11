# EXP-0313 — M25-H/I review hardening and P512 stall triage

**Status:** RETEST
**Milestone:** M25
**Date:** 2026-09-12
**Baseline commit:** `3fbe0f1`
**Candidate commit:** `f38745e` plus review hardening in the working tree

## Question

Are the qualified M25-H/I attention paths independently safe, and does the
retained M25-J source change explain the current wide P512 stall?

## Changes

The review fixes are deliberately limited to correctness and diagnostics:

* Mx attention Q8 workspace selection now follows FFN **or** O, so O-only is
  valid.
* Requested Mx FFN/O paths fail fast for unsupported Q4_K/Q6_K tensor types.
* Environment parsing for H/I/J requires exactly `0` or `1`.
* Device telemetry now separates allocation count, cumulative allocated bytes,
  live bytes, and peak live bytes; the CLI also reports `hipMemGetInfo` values.
* The attention bakeoff clears experimental environment variables before
  applying its selected mode and exposes `mx_ffn`, `mx_o`, and `mx_ffn_o` modes.

## Correctness

The Release build and all 24 CTest tests passed after the changes.

An invalid `MIINFER_MX_Q8_BATCH=banana` value was rejected with a nonzero
status and the expected strict-parser error.

The layer-3 B128 matrix completed with finite output and scalar parity in all
four configurations:

| FFN | O | mode | GPU time | live layer bytes | max abs | RMSE |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 0 | 0 | control | `19.724 ms` | `375,369,616` | `0.012` | `0.001` |
| 1 | 0 | H | `20.321 ms` | `329,076,624` | `0.555` | `0.010` |
| 0 | 1 | I | `20.159 ms` | `395,653,008` | `0.479` | `0.015` |
| 1 | 1 | H+I | `20.729 ms` | `346,853,264` | `0.751` | `0.017` |

This validates the previously untested O-only constructor/workspace path. It
is a B128 correctness matrix, not a performance qualification.

## P512 stall diagnostics

The exact full-model P512 control command reached prompt processing and then
failed to complete within the bounded run both with `MIINFER_PREFILL_MX_GDN=1`
and with it disabled. Both runs reported the expected control allocation of
`19,108,282,708 B`.

A matched B128 full-model diagnostic also stalled after prompt setup with both
the pre-J `3fbe0f1` binary and current main. Earlier M25-J diagnostics likewise
reproduced the stall with `MIINFER_MX_Q8_BATCH=0`. These observations do not
attribute the stall to M25-H/I or J.

A privileged GPU reset could not be performed because `sudo` required an
interactive password/TTY. The available telemetry showed no active KFD client
after cleanup, but `rocm-smi` reported the low-power warning and no useful GPU
activity during the stalled runs. Therefore this is not a clock-qualified
source regression result.

## Decision

**KEEP** the M25-H/I implementation as an opt-in qualified result; do not make
it the default yet. Keep M25-J opt-in and **RETEST** it later. The existing
`211.48 tok/s` H/I qualification remains the valid end-to-end result; this
experiment adds no new throughput claim.

## Follow-up

After a real cold GPU/device reset, run the same clean-shell script for
`3fbe0f1` and `f38745e` with control and H/I at P512, repeated three times.
Then add the same-process P512 → one-token continuation → P512 test before
any default promotion or further optimization.
