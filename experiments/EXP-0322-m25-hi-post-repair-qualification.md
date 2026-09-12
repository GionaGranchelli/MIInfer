# EXP-0322 — M25-H/I post-repair qualification

**Status:** KEEP; primary gate passed, stretch remains open  
**Milestone:** M25  
**Date:** 2026-09-12  
**Baseline:** `0d88b0a`  
**Reference:** pinned mx-llama `2e9d29fe736969160f17476ec6f0a6298cee6966`

## Hypothesis

The resident-M23 repair in `1bce8cb` preserves the qualified H/I P512 speed
while making continuation and repeated same-process prefill safe.

## Environment

- AMD Instinct MI50 / gfx906 / Wave64
- Qwen3.8-27B-Q4_K_M, SHA-256
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- Release build, target `gfx906`, GNU 16.2.1 / HIP Clang 20.0.0
- exact 512-token repeated-fox prompt
- clean `env -i` process environment
- SCLK/MCLK `1606/1000 MHz`
- `MIINFER_MX_Q8_BATCH=0`; `MIINFER_MX_PIPELINE` unset
- qualified base vector from EXP-0314 plus
  `MIINFER_PREFILL_WIDE_MX_REPACKED_ATTN_FFN=1` and
  `MIINFER_PREFILL_WIDE_MX_REPACKED_ATTN_O=1`

The six P512 samples ran as three fresh processes with the real continuation
and same-process repeat check. Continuous telemetry retained 479 samples in
`/tmp/mi50-hi-postfix-l2ZHtX`.

## Correctness

- H/I matrix `00/10/01/11` passed at B128 and B512; every output was finite.
- Same-process `P512 → continuation → P512` passed three times. The first
  token was `13477` (`brown`) and the continuation token was `37550`.
- Release CTest passed `24/24`.
- 128-token no-stream generation completed without crash, nonfinite output, or
  delayed state failure.

## Results

| P512 latency (ms) | throughput (tok/s) |
| ---: | ---: |
| 2516.37 | 203.47 |
| 2511.98 | 203.82 |
| 2497.32 | 205.02 |
| 2442.26 | 209.64 |
| 2503.92 | 204.48 |
| 2487.38 | 205.84 |

Sorted median latency is `(2497.32 + 2503.92) / 2 = 2500.62 ms`, or
`204.75 tok/s`. This clears the 200 tok/s primary gate. The historical
pre-repair peak `211.48 tok/s` is retained as a separate result and is not
used to qualify the repaired path.

The pinned reference is `2317.872 ms` / `220.892 tok/s`; repaired H/I remains
`182.748 ms` slower, or `7.89%` behind on latency.

The matched 128-token decode check measured `501.188 ms/token` for H/I and
`501.951 ms/token` for control. H/I showed no decode regression.

## Hardware and memory

All 479 telemetry samples reported SCLK/MCLK `1606/1000 MHz`; temperature was
`32–40 C` and observed power was `28–191 W`.

H/I live, total, and peak MIInfer allocation at setup were
`21,993,242,964 B`; `hipMemGetInfo` reported `11,484,004,352 B` free of
`34,342,961,152 B`. The matched control used `19,108,282,708 B` and left
`14,453,571,584 B` free. The extra H/I resident-M23 FFN representation is
therefore measurable but did not reduce decode correctness or measured decode
latency.

## Profiling

The post-repair operator profile is diagnostic only because event profiling
adds wall overhead. It reports zero hot-path weight uploads. The selected wide
sample attributes approximately:

| recurrent family | GPU ms |
| --- | ---: |
| GDN core | 168.035 |
| SSM output + residual | 291.556 |
| FFN gate/up + SwiGLU | 1572.790 |
| FFN down + residual | 786.796 |

The next candidate must target the recurrent FFN execution contract; another
GDN or attention micro-tuning pass is not justified by this profile.

## Preset verification

`MIINFER_PRESET=m25_hi_qualified` now clears ambient `MIINFER_*` selectors
(preserving `MIINFER_API_KEY`), applies the vector above, and prints the
resolved values at startup. A clean preset run passed the same-process check:

```text
same_process_p512_check=PASS first_token=13477(brown) continuation_token=37550
first_prefill_ms=2476.79 continuation_ms=516.59 repeat_prefill_ms=2454.05
```

An unknown preset fails before model loading with exit status 2.

## Decision

**KEEP** H/I as the qualified opt-in path. The primary performance and
correctness gates now pass after the repair. Do not claim a repaired-path
stretch match: it remains below the pinned reference.

## Follow-up

Use the versioned `m25_hi_qualified` preset for future qualification runs. Then
choose one recurrent FFN differential experiment with a measured
production-shape baseline; do not add another speculative kernel.
