# EXP-0304 — M25-C mx register-pipeline transplant

**Status:** REJECT  
**Milestone:** M5/M6  
**Date:** 2026-09-11  
**Baseline commit:** `89689c7`  
**Candidate commit:** `8efeaca`  

## Question

Does porting the pinned mx register-to-LDS software pipeline improve the
compact MI50 MMQ kernel over the simpler staged implementation?

## Hypothesis

Prefetching the next Q4_K/Q5_K/Q6_K weight tile and Q8_1 input tile into
registers while the current tile computes should hide global-memory latency,
as it does in the pinned mx kernel.

## Source / prior art

- `mxxm-t/mx-llama.cpp`, exact checkout
  `2e9d29fe736969160f17476ec6f0a6298cee6966`
- Relevant source: `ggml/src/ggml-cuda/mmq.cuh` and
  `ggml/src/ggml-cuda/q8_repack/repack-kernels.cuh`
- Upstream license: MIT, copyright `2023-2026 The ggml authors`; see
  `docs/references.md`

## Baseline

The baseline is the simpler MIInfer compact staged kernel from `89689c7`:
each 64-row × 4-subblock weight tile and 128-token input tile is staged in
shared memory before the dot-product loop.

## Candidate

The candidate adds a full-tile fast path with register-held next weight and
input values, explicit shared-memory staging, and a compute/store/barrier
pipeline. It is available for isolated retests with:

```sh
MIINFER_MX_PIPELINE=1 ./build/mi50-release/miinfer-m24-projection-bakeoff ...
```

The measured staged kernel remains the default because the environment flag
is opt-in.

## Environment

- AMD Instinct MI50 / gfx906, 60 CUs, Wave64
- Qwen3.8-27B-Q4_K_M.gguf, SHA-256
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- Linux kernel `7.1.10-200.fc44.x86_64`
- ROCm/HIP `7.1.52802-9999`, clang `20.0.0.rocm`
- observed clocks: SCLK 1606 MHz, MCLK 1000 MHz

## Benchmark

Command shape, with `MIINFER_MX_PIPELINE` unset for the baseline and set to
`1` for the candidate:

```sh
./build/mi50-release/miinfer-m24-projection-bakeoff MODEL gate q4 0
./build/mi50-release/miinfer-m24-projection-bakeoff MODEL ssm_out q5 0
./build/mi50-release/miinfer-m24-projection-bakeoff MODEL down q6 0
```

Each run used B512 and seven measured iterations after two warmups. Fresh
median kernel timings were:

| projection | staged default (us) | register pipeline (us) | candidate / baseline |
|---|---:|---:|---:|
| Q4 gate | 7,871.833 | 18,051.664 | 2.29x slower |
| Q5 ssm_out | 3,150.236 | 6,763.834 | 2.15x slower |
| Q6 down | 8,123.994 | 18,275.188 | 2.25x slower |

## Correctness

The Q4/Q5/Q6 CPU contract checks passed with the same maximum absolute errors
as EXP-0303: `6.0e-7`, `1.67e-6`, and `4.8e-7`. Primitive outputs were finite.

## Results

The transplant is a large regression across all three tested quantizations.
The added register-held values, shared scale/input planes, and pipeline
control did not recover their intended latency hiding on this MI50 build.
No hardware-counter capture was taken, so the exact contribution of VGPR
pressure, instruction scheduling, occupancy, or compiler lowering is not
isolated by this experiment.

## Decision

**REJECT as the default kernel.** Keep the implementation behind
`MIINFER_MX_PIPELINE=1` only as a reproducible retest point; do not claim the
mx software pipeline ports usefully without profile evidence.

## Follow-up

If this path is revisited, capture VGPR/LDS/occupancy and instruction metrics
first, then test one pipeline change at a time. The compact layout and staged
kernel remain the production candidate.
