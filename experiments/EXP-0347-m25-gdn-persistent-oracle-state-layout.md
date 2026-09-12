# EXP-0347 — M25 persistent oracle GDN state layout

**Status:** REJECT; retained opt-in for contract reproducibility  
**Milestone:** M25 stretch investigation  
**Date:** 2026-09-12  
**Baseline:** `a24b8d5`  
**Reference:** `mx-llama.cpp@2e9d29fe736969160f17476ec6f0a6298cee6966`

## Question

Does making the oracle's `[value][key]` recurrent-state layout persistent
remove enough state traffic from the Mx GDN scan to improve P512 latency?

## Hypothesis

The current Mx prefill scan uses MIInfer's persistent `[key][value]` layout,
while the pinned oracle scans a coalesced `[value][key]` layout. The earlier
temporary conversion candidate paid for two full-state transposes per
recurrent layer and was slightly slower. Using the oracle layout as the
persistent state should preserve its coalesced loads/stores without those
transposes.

## Candidate

`MIINFER_PREFILL_MX_GDN_EXTERNAL_STATE=1` selects the adapted Mx scan's
`[value][key]` state indexing. It requires
`MIINFER_DELTA_TRANSPOSED_STATE=0`, so the existing non-transposed decode
state-update kernels consume the same persistent layout. The qualified H/I
path, weights, Q8 path, and decode selector remain unchanged by default.

The candidate adds no persistent allocation: its state is the existing
`48 * 128 * 128` float buffer. The earlier out-of-place state contract remains
rejected by EXP-0343.

## Environment

- AMD Instinct MI50 / gfx906 / Wave64
- Qwen3.8-27B-Q4_K_M; SHA-256
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- exact 512-token repeated-fox prompt, context capacity `1024`
- release binary, fixed SCLK/MCLK `1606/1000 MHz`
- clean `env -i` processes, explicit H/I vector, `MIINFER_MX_Q8_BATCH=0`
- three fresh-process candidate/control pairs, interleaved
- candidate additionally set:
  `MIINFER_PREFILL_MX_GDN_EXTERNAL_STATE=1`
  `MIINFER_DELTA_TRANSPOSED_STATE=0`
- control omitted both candidate selectors and retained the default
  transposed state layout

## Correctness

The candidate passed the real same-process state gate:

```text
same_process_p512_check=PASS first_token=13477(brown)
continuation_token=37550 first_prefill_ms=2527.64
continuation_ms=494.601 repeat_prefill_ms=2543.1
```

All six timing processes processed exactly 512 prompt tokens and completed
finite output. Candidate and control reported the same live allocation:
`21,993,242,964 B`.

## Results

| path | P512 samples (ms) | median ms | median tok/s |
| --- | --- | ---: | ---: |
| persistent oracle state candidate | 2502.69, 2484.82, 2815.64 | **2502.69** | **204.58** |
| H/I control | 2474.03, 2494.44, 2506.85 | **2494.44** | **205.26** |

The candidate is `8.25 ms` slower by median, or `+0.331%`. The `2815.64 ms`
candidate sample is retained as a raw outlier; it was not discarded to make
the result look better.

## Interpretation

The persistent layout removes the temporary transpose cost and preserves the
oracle's state indexing, but the resulting scan does not beat the existing
MIInfer state contract end to end. The state-layout differential is therefore
not the missing P512 stretch gap under the current gfx906 kernels. The
candidate also changes the decode state-update family, so its passing
continuation gate is not evidence of a decode performance win.

## Decision

**REJECT for qualification and default promotion.** Retain the selector as a
small, reproducible external-contract probe, but do not add it to
`m25_hi_qualified` or pursue more GDN state-layout variants without a new
positive differential.

## Follow-up

Stop isolated GDN state-layout work. The remaining stretch investigation must
return to a like-for-like whole recurrent execution differential, especially
the QKV/GDN materialization and composition boundaries.
