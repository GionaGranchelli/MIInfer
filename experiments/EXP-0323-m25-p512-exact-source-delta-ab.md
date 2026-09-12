# EXP-0323 — M25 P512 exact pre-J/source-delta A/B

**Status:** KEEP; M25-J source delta not implicated
**Milestone:** M25
**Date:** 2026-09-12
**Baseline commit:** `3fbe0f1`
**Candidate commit:** `f38745e`

## Question

Does the retained M25-J source delta explain the previously observed P512
stall when the qualified H/I path and control are run from clean processes?

## Environment

- AMD Instinct MI50 / gfx906 / Wave64
- Qwen3.8-27B-Q4_K_M; model SHA-256
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- Release builds at the two stated commits
- context capacity `1024`, exact 512-token prompt made by repeating
  `The quick brown fox jumps over the lazy dog. ` 51 times and appending
  `The quick`
- `--max-tokens 0 --no-stream`
- each process launched with `env -i`, the same explicit runtime vector, and
  `MIINFER_MX_Q8_BATCH=0`; `MIINFER_MX_PIPELINE` was unset
- three fresh processes per cell, in interleaved control/H/I order within each
  commit
- continuous `sample-gpu.sh` telemetry: 1541 samples, all SCLK/MCLK
  `1606/1000 MHz`

The H/I vector was:

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
```

The control omitted the final two H/I selectors.

## Correctness

All 12 fresh processes exited successfully, processed exactly 512 prompt
tokens, and completed finite P512 execution without a stall or runtime error.
The existing H/I continuation, matrix, CTest, and long-generation evidence
remains covered by EXP-0322.

## Results

| commit | path | raw P512 ms | median ms | median tok/s | device bytes |
| --- | --- | --- | ---: | ---: | ---: |
| `3fbe0f1` | control | 2915.72, 2853.56, 3310.51 | 2915.72 | 175.60 | 19,108,282,708 |
| `3fbe0f1` | H/I | 2458.09, 2469.34, 2487.13 | 2469.34 | 207.34 | 18,472,649,044 |
| `f38745e` | control | 3019.78, 2886.77, 2873.97 | 2886.77 | 177.36 | 19,108,282,708 |
| `f38745e` | H/I | 2462.20, 2408.38, 2495.80 | 2462.20 | 207.94 | 18,472,649,044 |

Relative to pre-J, the candidate commit is `28.95 ms` faster on the control
median and `7.14 ms` faster on the H/I median. Neither comparison indicates a
J-induced regression; H/I remains about `14.7%` faster than its matched
control on the candidate commit.

## Interpretation

The retained M25-J source change is not sufficient to explain the earlier
stall. The exact source-delta A/B reproduces the expected H/I speedup on both
commits, with stable clocks and no incomplete run. This rules out a
reproducible P512 source regression from J under `MIINFER_MX_Q8_BATCH=0`.

The run used clean process/device access but did not perform a privileged
hardware reset; the earlier stall therefore remains best classified as
transient device/runtime or harness state rather than source-attributed.

## Decision

**KEEP** M25-H/I as the qualified opt-in path. **KEEP** M25-J disabled for
performance purposes. Do not add another optimization based on the old stall.
The primary 200 tok/s gate is met; the pinned reference stretch target remains
open.

## Follow-up

Stop speculative kernel work until a new production-shape differential is
measured. The next permitted performance change must target the profiled
recurrent FFN orchestration gap or port a complete, contract-compatible
external execution path with correctness and A/B evidence.
