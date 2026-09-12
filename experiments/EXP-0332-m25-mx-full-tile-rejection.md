# EXP-0332 — Mx MMQ full-tile specialization

## Hypothesis

The exact P512 Mx MMQ geometry is always a full `64 x 128` tile. A
compile-time fast path that removes row and token bounds checks, matching the
pinned mx-llama full-tile branch, should improve the default staged kernel.

## Motivation

The pinned source has separate checked and unchecked MMQ paths. MIInfer's
default legacy Mx kernel only had the checked form. This is a narrow control
flow port with no layout or arithmetic change.

## Baseline

The committed staged Mx MMQ kernel with bounds checks in every weight, input,
and output tile operation.

## Candidate

Temporarily dispatch a `FULL_TILE` kernel specialization whenever rows are a
multiple of 64 and the token count is a multiple of 128. Partial tiles retain
the existing checked kernel. The candidate was removed after screening.

## Environment

- AMD Instinct MI50 / gfx906 / Wave64
- Qwen3.8-27B-Q4_K_M, exact repeated-fox P512 prompt
- Release binaries, explicit M25-H/I vector, clean `env -i` process
- `MIINFER_MX_Q8_BATCH=0`; `MIINFER_MX_PIPELINE` unset
- Same allocation and VRAM telemetry as the control

## Benchmark

Three interleaved fresh-process pairs in order `C A C A C A`, with
`--max-tokens 0 --no-stream`. This was a screening series, not a promotion
qualification.

## Correctness

All six P512 processes completed without a HIP fault or non-finite runtime
failure. No continuation check was run because the candidate failed the
screening stability gate; no correctness claim is made beyond process
completion.

## Results

| order | path | P512 ms | tok/s |
| ---: | --- | ---: | ---: |
| 1 | control | 2503.23 | 204.54 |
| 2 | candidate | 2456.62 | 208.42 |
| 3 | control | 2527.38 | 202.58 |
| 4 | candidate | 2735.09 | 187.20 |
| 5 | control | 2842.74 | 180.11 |
| 6 | candidate | 2420.50 | 211.53 |

The paired deltas were `-46.61`, `+207.71`, and `-422.24 ms`; the candidate
range was `314.59 ms`. Control sample 5 was an obvious contaminated outlier,
but excluding it still leaves an unstable candidate with one large regression.
The all-sample medians are not a qualified comparison because both sides are
dominated by this short, high-variance screen.

## Profiling

No profiler run was justified. The expected branch-removal gain was not
separable from runtime-state variance.

## Decision

**REJECT.** Restore the single checked staged kernel and do not add a full-tile
selector without a stable five-plus-sample result.

## Follow-up

The pinned geometry is already represented, but its control-flow specialization
is not currently justified as a production change. The P512 stretch remains
open.
