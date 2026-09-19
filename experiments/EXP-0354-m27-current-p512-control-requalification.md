# EXP-0354 — Current HEAD P512 control requalification

**Status:** control baseline retained for EXP-0355
**Milestone:** M27 cold-prefill continuation
**Date:** 2026-09-19
**Baseline commit:** `12fc127` (`exp: qualify live Pi prefix reuse path`)
**Reference:** `mx-llama.cpp@2e9d29fe736969160f17476ec6f0a6298cee6966`

## Question

What is the present six-pair P512 baseline on the committed current HEAD before
retesting the retained Mx attention O/FFN decode-reuse representation?

## Environment and workload

- AMD Instinct MI50 / gfx906 / Wave64
- Qwen3.8-27B-Q4_K_M, SHA-256
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- AMD HIP `7.1.52802-9999`, HIP Clang `20.0.0.rocm`
- SCLK/MCLK fixed at `1606/1000 MHz`
- six interleaved fresh-process pairs, mx first in each pair
- no profiler; zero generated tokens; model loaded per process
- MIInfer `MIINFER_PRESET=m25_hi_qualified`
- prompt: `'The quick brown fox jumps over the lazy dog. ' * 51 + 'The quick'`,
  exactly 512 tokens in MIInfer
- pinned `llama-bench -p 512 -n 0 -r 1`; its P512 prompt tokens are synthetic,
  since that benchmark does not accept the exact MIInfer prompt text
- raw logs and telemetry: `/tmp/m27-p512-current-7bfe0b3/`

The clean-process commands matched EXP-0348, using the current `miinfer`
binary and the same pinned `llama-bench` build. Telemetry ran concurrently at
250 ms intervals using `scripts/sample-gpu.sh`.

## Results

| pair | mx ms | mx tok/s | MIInfer ms | MIInfer tok/s | MIInfer allocation B |
|---:|---:|---:|---:|---:|---:|
| 1 | 2307.142 | 221.920 | 2745.65 | 186.48 | 21,993,243,028 |
| 2 | 2305.579 | 222.070 | 2382.43 | 214.91 | 21,993,243,028 |
| 3 | 2307.087 | 221.925 | 2525.33 | 202.75 | 21,993,243,028 |
| 4 | 2306.815 | 221.951 | 2490.38 | 205.59 | 21,993,243,028 |
| 5 | 2305.455 | 222.082 | 2407.61 | 212.66 | 21,993,243,028 |
| 6 | 2306.797 | 221.953 | 2482.99 | 206.20 | 21,993,243,028 |
| **median** | **2306.806** | **221.952** | **2486.685** | **205.897** | **21,993,243,028** |

Current median gap: `179.879 ms/P512`, or `7.798%` relative to mx latency.
All MIInfer processes processed exactly 512 prompt tokens and emitted finite
output. No samples were removed; the `2745.65 ms` first MIInfer result remains
in the median calculation.

## Hardware audit

The continuous capture contained `1348` samples. Every sample held SCLK/MCLK
at `1606/1000 MHz`. Junction temperature ranged from `34 C` to `60 C`; package
power averaged `39.9 W` and peaked at `191 W` against the `225 W` cap. Peak
reported VRAM use was `22,945,673,216 B`. MIInfer setup allocation and free
memory were constant across all six runs: `21,993,243,028 B` allocated and
`11,484,004,352 B` free.

## Interpretation

The current control is slightly faster than EXP-0348's `2499.345 ms` median,
but still below 210 tok/s. EXP-0355 contains the matched six-pair test of the
lower-footprint candidate. The oracle uses synthetic prompt tokens, so the
absolute cross-runtime differential retains that known caveat.
