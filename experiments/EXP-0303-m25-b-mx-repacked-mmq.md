# EXP-0303 — M25-B mx compact repacked MMQ

**Status:** RETEST  
**Milestone:** M5/M6  
**Date:** 2026-09-11  
**Baseline commit:** `1c4e3d0`  
**Candidate commit:** `dc938c4`

## Question

Does porting the pinned mx `q8_repack` Q4_K/Q5_K/Q6_K storage and Q8_1
activation contracts reduce MIInfer's recurrent wide-prefill cost on MI50?

## Hypothesis

The compact row-major planes and matching Wave64 MMQ tile will reduce global
weight traffic and dequantization work versus the existing M23 expanded tile.

## Source / prior art

- `mxxm-t/mx-llama.cpp`, exact checkout
  `2e9d29fe736969160f17476ec6f0a6298cee6966`
- Source files: `ggml/src/ggml-cuda/q8_repack/README.md`,
  `repack-common.cuh`, `repack-common.cu`, `repack-kernels.cuh`, `mmq.cuh`,
  and `quantize.cu`
- Upstream license: MIT, copyright `2023-2026 The ggml authors`; see
  `docs/references.md`

The port reimplements the layout and arithmetic contract in MIInfer. It does
not add mx as a dependency or copy the generic llama.cpp execution runtime.

## Baseline

The M23 expanded resident tile and M23 Q8_1 block were used for the control.
The exact model workload was `blk.0.ffn_gate.weight` Q4_K,
`blk.0.ssm_out.weight` Q5_K, and `blk.0.ffn_down.weight` Q6_K at B64/B128/B256/B512.

## Candidate

The candidate adds:

- compact mx-compatible Q4_K, Q5_K, and Q6_K host repacking;
- the 144-byte mx Q8_1 block (`ds4` for affine Q4/Q5 and `d4` for Q6);
- a Wave64 64-row × 128-token MMQ kernel;
- an opt-in recurrent wide-prefill and resident-all path controlled by
  `MIINFER_PREFILL_WIDE_MX_REPACKED_MMQ=1`.

The existing M23 path, attention path, and resident M23 decode path remain
available. The opt-in path uses the compact contract for recurrent decode when
resident-all is enabled.

## Environment

- AMD Instinct MI50 / gfx906, 60 CUs, Wave64
- Qwen3.8-27B-Q4_K_M.gguf, SHA-256
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- Linux kernel `7.1.10-200.fc44.x86_64`
- ROCm/HIP `7.1.52802-9999`, clang `20.0.0.rocm`
- observed clocks: SCLK 1606 MHz, MCLK 1000 MHz
- observed spot state after runs: 37 C edge / 39 C junction / 36 C memory

These runs had spot clock checks, not continuous telemetry, so the runtime
timings are not final clock-qualified claims.

## Primitive benchmark

Command:

```sh
./build/mi50-release/miinfer-m24-projection-bakeoff MODEL gate q4 0
./build/mi50-release/miinfer-m24-projection-bakeoff MODEL ssm_out q5 0
./build/mi50-release/miinfer-m24-projection-bakeoff MODEL down q6 0
```

Median kernel microseconds from seven measured iterations after two warmups:

| projection | batch | M23 MMQ | mx compact MMQ | M23 / mx | mx bytes |
|---|---:|---:|---:|---:|---:|
| Q4 gate | 512 | 19,207.823 | 7,852.154 | 2.45x | 50,413,568 |
| Q5 ssm_out | 512 | 7,120.314 | 3,144.957 | 2.26x | 21,708,800 |
| Q6 down | 512 | 16,531.029 | 8,149.594 | 2.03x | 73,195,520 |

The corresponding M23 resident sizes were 72,417,280, 25,559,040, and
77,987,840 bytes. The mx CPU contract checks passed with maximum absolute
errors of `6.0e-7`, `1.67e-6`, and `4.8e-7` for Q4/Q5/Q6.

## Runtime A/B

The exact 512-token repeated fox prompt used the existing layer-major,
resident-all configuration. The candidate additionally set
`MIINFER_PREFILL_WIDE_MX_REPACKED_MMQ=1`; the old M23 repack flag remained set
so the unchanged attention layers used their established path.

| path | P512 samples (ms) | median tok/s | allocation |
|---|---:|---:|---:|
| M23 control | 5,120.27; 5,122.34 | 99.98 | 22,801,772,884 B |
| Mx recurrent candidate | 3,617.26; 4,001.63 | 134.39 | 19,108,282,708 B |

The candidate's median is a `1.34x` speedup and reduces measured prefill
latency by `25.6%` in this small A/B series. A P512 + one-token candidate run
also completed with 3.41 ms decode time.

One earlier candidate setup omitted `MIINFER_PREFILL_WIDE_REPACKED_MMQ`,
causing attention fallback and an 8,469.75 ms result. That run is invalid for
this comparison and is retained here only to document the configuration trap.

## Correctness

- compact Q4/Q5/Q6 primitive outputs were finite;
- CPU contract checks passed as above;
- Mx P512 prefill and one-token decode completed with finite output;
- greedy short-prompt output matched the M23 control (`The answer is no`).

Full intermediate-tensor parity and long-generation parity are still required.

## Interpretation

The compact contract is a strong primitive-level win and the first integrated
recurrent P512 result materially closes the M23 gap. The memory reduction is
also favorable. The candidate currently ports the storage/arithmetic contract
but not every mx software-pipeline optimization, so further porting should be
guided by runtime profiling rather than assumed.

## Decision

**KEEP as the opt-in recurrent prefill candidate; RETEST for qualification.**
Do not make it the default until continuous hardware telemetry, repeated
interleaved A/B runs, intermediate parity, and longer generation checks pass.

## Follow-up

1. Repeat the runtime A/B with continuous SCLK/MCLK/power capture.
2. Compare intermediate recurrent tensors and logits against the M23 control.
3. Profile the compact kernel, then port only the measured mx software-pipeline
   pieces that address the remaining bottleneck.
