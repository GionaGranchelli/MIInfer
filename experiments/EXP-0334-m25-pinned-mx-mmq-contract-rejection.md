# EXP-0334 — M25 complete pinned Mx MMQ contract rejection

## Hypothesis

Porting the pinned `mx-llama.cpp` large-batch q8-repack MMQ contract as one
unit will beat MIInfer's staged gfx906 Mx kernel at the recurrent FFN and
attention production shapes.

## Motivation

The isolated pinned schedule components had already failed independently:
launch bounds, metadata hoisting, register prefetch, full-tile bounds removal,
and the vectorized epilogue. The project rule and the current stretch goal
called for one complete contract port before abandoning the external path.

## Baseline

MIInfer commit `5457683`, with the measured staged Mx MMQ kernel selected by
default.

## Candidate

An opt-in MIInfer-owned port of the pinned `mmq_gemm_repacked` large-batch
contract, selected by `MIINFER_MX_PINNED=1`. It preserves the pinned
register-prefetch ownership, metadata grouping, X-word hoisting,
full-tile/checked-tile split, `__launch_bounds__(256, 2)`, and float4 epilogue
for the existing MIInfer Q4/Q5/Q6 repacked layouts. The source was adapted
from mx-llama.cpp commit
`2e9d29fe736969160f17476ec6f0a6298cee6966` under its existing MIT provenance.

## Environment

- AMD Instinct MI50 / gfx906 / Wave64
- Qwen3.8-27B-Q4_K_M GGUF, SHA-256
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- Release build, same binary except for the opt-in candidate
- exact 512-token repeated-fox prompt
- context capacity `1024`
- clean `env -i` processes with the explicit M25 H/I vector
- `MIINFER_MX_Q8_BATCH=0`; `MIINFER_MX_PIPELINE` unset
- five fresh interleaved pairs in `control, pinned` order
- continuous `scripts/sample-gpu.sh` telemetry at 250 ms: 1516 samples
- all telemetry samples reported SCLK/MCLK `1606/1000 MHz`
- maximum observed junction temperature `57 C`
- maximum observed VRAM use `22,967,218,176 B`

The baseline executable was saved from commit `5457683` before enabling the
candidate. Both paths reported `21,993,242,964 B` runtime allocation.

## Benchmark

```text
--max-tokens 0 --no-stream
```

## Correctness

The candidate completed all five P512 runs with finite output. A separate
real same-process check passed the continuation and repeat-P512 state gate:

```text
first_token=13477(brown)
continuation_token=37550
```

## Results

| path | P512 samples (ms) | median ms | median tok/s |
| --- | --- | ---: | ---: |
| staged control | 2508.21, 2492.78, 2476.75, 2515.02, 2498.36 | 2498.36 | 204.93 |
| complete pinned contract | 2859.98, 2872.76, 2723.41, 2876.76, 2847.76 | 2859.98 | 179.02 |

The complete port was `361.62 ms` slower by median (`14.47%`) and `12.16%`
slower in throughput. The candidate used the same tracked allocation.

## Profiling

No kernel profiler was needed after the clock-qualified end-to-end result:
the complete contract was consistently slower across five fresh pairs. The
compiler also reported final occupancy one for the Q4/Q5 candidate despite
the requested two-block launch bound, matching the earlier isolated warning.

## Interpretation

The pinned MMQ schedule is not transferable to MIInfer's exact gfx906
production workload as a complete unit. The external result likely depends on
its surrounding runtime, launch mix, or shape distribution; copying its inner
contract does not reproduce its performance here.

## Decision

**REJECT for default and qualified preset.** Keep the candidate opt-in for
provenance and possible future retest, but leave MIInfer's staged one-block
kernel selected by default. Do not pursue more isolated transplants of this
MMQ contract without a new measured production-shape reason.

## Re-evaluation

A one-screen rescue test changed only the candidate's launch annotation to
`__launch_bounds__(256, 1)`, leaving the complete schedule unchanged. It
measured `2980.35 ms` (`171.79 tok/s`) on the same exact P512 path, so the
regression is not explained by the pinned two-block occupancy request. That
temporary isolation hook was removed.

## Follow-up

Return the stretch investigation to the measured recurrent FFN tail and its
launch/orchestration gap. The pinned MMQ contract has now been tested both by
component and as a complete port.
