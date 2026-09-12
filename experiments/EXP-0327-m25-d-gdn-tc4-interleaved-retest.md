# EXP-0327 — M25-D GDN Tc4 interleaved qualification retest

## Hypothesis

The M25-D four-column GDN shard improves end-to-end P512 when machine state is
controlled, and is safe to promote into the qualified M25 preset.

## Baseline

MIInfer commit `40f2373`, explicit qualified M25 H/I vector, with
`MIINFER_MX_GDN_TC4=0` selecting `mx_gdn_chunk_kernel<128, 64, 2>`.

## Candidate

The same commit and vector with `MIINFER_MX_GDN_TC4=1` selecting
`mx_gdn_chunk_kernel<128, 64, 4>`.

## Environment

- AMD Instinct MI50 / gfx906 / Wave64
- Qwen3.8-27B-Q4_K_M GGUF, SHA-256
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- Release build; ROCm `7.1.52802-9999`; clang `20.0.0.rocm`
- context capacity `1024`; exact 512-token repeated-fox prompt
- clean `env -i` process with the explicit M25 H/I vector
- `MIINFER_MX_Q8_BATCH=0`; `MIINFER_MX_PIPELINE` unset
- continuous `scripts/sample-gpu.sh` telemetry at 250 ms

## Benchmark

Six fresh processes ran in interleaved order `0, 1, 0, 1, 0, 1` with
`--max-tokens 0 --no-stream`. The full vector was:

```text
MIINFER_CONTEXT_CAPACITY=1024
MIINFER_PREFILL_LAYER_MAJOR=1
MIINFER_PREFILL_WIDE_CHUNK=1
MIINFER_PREFILL_FULL_LAYER_MAJOR=1
MIINFER_PREFILL_WIDE_ATTN=1
MIINFER_PREFILL_WIDE_REPACKED_MMQ=1
MIINFER_PREFILL_WIDE_MMQ_QKV=1
MIINFER_PREFILL_WIDE_MMQ_SSM_OUT=1
MIINFER_PREFILL_WIDE_MMQ_FFN=1
MIINFER_M23_REPACKED_ROW128=1
MIINFER_PREFILL_REPACKED_RESIDENT_ALL=1
MIINFER_PREFILL_CHUNK=512
MIINFER_PREFILL_MX_GDN=1
MIINFER_PREFILL_WIDE_MX_REPACKED_MMQ=1
MIINFER_MX_Q8_BATCH=0
MIINFER_PREFILL_WIDE_MX_REPACKED_ATTN_FFN=1
MIINFER_PREFILL_WIDE_MX_REPACKED_ATTN_O=1
MIINFER_MX_GDN_TC4=0|1
```

## Correctness

All six processes completed finite P512 execution. The standalone GDN oracle,
full CTest, and same-process continuation/repeat-P512 checks remain passed
from EXP-0326.

## Results

| path | P512 ms | tok/s |
| --- | ---: | ---: |
| Tc2 | 2412.10, 2500.43, 2449.30 | median `209.04` |
| Tc4 | 2489.16, 2473.73, 2489.42 | median `205.69` |

Tc4 was `1.63%` slower by median. Every telemetry file observed only
`1606/1000 MHz` SCLK/MCLK; maximum junction temperature was `58 C`, and peak
observed VRAM use was approximately `22.945 GB`.

## Profiling

No kernel profiler was needed: the A/B was clock-qualified and the candidate
direction was negative. The earlier isolated GDN improvement does not survive
this end-to-end interleaved qualification.

## Interpretation

The isolated Tc4 mapping is numerically valid and can still be useful as an
opt-in kernel experiment, but its production benefit is not stable enough for
the qualified preset. The earlier `215.85 tok/s` pair was machine-state spread,
not a promotion-quality result.

## Decision

**REJECT for default promotion; KEEP opt-in.**

## Follow-up

Do not spend another optimization pass on the same Tc4 geometry. A future
stretch attempt must target a different measured production bottleneck or port
a materially different gfx906 execution contract from the pinned reference.
