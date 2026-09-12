# EXP-0333 — Q4/Q5-only Mx MMQ occupancy annotation

## Hypothesis

The dominant affine Q4/Q5 Mx projections may benefit from the pinned
`__launch_bounds__(256, 2)` annotation even though the earlier all-type test
regressed when Q6 used the same occupancy target.

## Motivation

This isolates the occupancy question to the recurrent Q4 FFN and attention
FFN/O work while retaining the known one-block Q6 Down/SSM path.

## Baseline

The committed Mx MMQ kernel uses `__launch_bounds__(256, 1)` for Q4, Q5, and
Q6. EXP-0315 measured the all-type two-block annotation at `2679.44 ms` /
`191.08 tok/s` versus `2435.75 ms` / `210.20 tok/s` control.

## Candidate

Temporarily changed the template annotation to two resident blocks for Q4 and
Q5, while retaining one block for Q6. No other source or runtime setting
changed. The candidate was removed after screening.

## Environment

- AMD Instinct MI50 / gfx906 / Wave64
- Qwen3.8-27B-Q4_K_M, exact repeated-fox P512 prompt
- Release build, explicit M25-H/I vector, clean `env -i` process
- `MIINFER_MX_Q8_BATCH=0`; `MIINFER_MX_PIPELINE` unset

## Benchmark

One fresh-process P512 screen with `--max-tokens 0 --no-stream`. This was not
a clock-qualified or promotion comparison.

## Correctness

The candidate completed finite P512 execution without a runtime error. No
continuation correctness claim is made from this screening run.

## Results

| path | P512 ms | tok/s |
| --- | ---: | ---: |
| Q4/Q5 two-block annotation | 2533.39 | 202.10 |

The sample did not beat the established one-block control range and provided
no reason to reopen the rejected all-type occupancy experiment.

## Decision

**REJECT.** Restore `__launch_bounds__(256, 1)` for all Mx MMQ types.

## Follow-up

Do not tune occupancy without VGPR/LDS evidence. The P512 stretch remains open
and the qualified kernel is unchanged.
