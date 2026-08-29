# EXP-0003 — FP16 GEMV Bottleneck Characterization

**Status:** RETEST  
**Milestone:** M1 — Kernel Laboratory  
**Date:** 2026-08-29  
**Baseline commit:** `b7dfbddce33f2dfd6d006bfe198a02a6f27dbf7f`  
**Diagnostic tooling:** `345ff89`

## Question

What limits the accepted MIInfer FP16 GEMV baseline in the K/V, Q/O, G/U,
and D regimes on the MI50, and which path is the real FP16 decode competitor
in the pinned gfx906 llama.cpp reference?

## Hypothesis and scope

The 8 MiB K/V result is limited by insufficient memory-level parallelism/grid
scale rather than by the MI50's practical streaming bandwidth. The pinned
reference is expected to use its custom `mmvf` FP16 vector kernel rather than
the tested hipBLAS GEMM-N=1 path.

This is an observation experiment. EXP-0002 kernel semantics were not changed,
and no gfx906 optimization candidate, runtime, model loader, quantized kernel,
or inference functionality was added. The copy and reduction programs are
explicitly diagnostic only.

## Baseline and controls

The baseline is EXP-0002's row-major FP16 `M x K` kernel: one 256-thread
workgroup per row, FP32 per-thread accumulation, 256-float LDS tree reduction,
and FP16 output. The accepted implementation commit is
`b7dfbddce33f2dfd6d006bfe198a02a6f27dbf7f`; accepted results are
`a82cd0d81d140d804c470d1d47c6f185e9b46367`.

All runs used the M0-approved state: gfx802/Tonga unbound from KFD,
`rocminfo` reporting gfx906, MIInfer validation passing, and no competing
Ollama/llama workload. The final diagnostic release build reports
`git_commit=f6963a82968f`, `git_dirty=false` before the later diagnostic
integration commit.

Raw result, environment, and telemetry artifacts are retained under:

```text
bench/results/EXP-0003-memory/
bench/results/EXP-0003-sweeps/
bench/results/EXP-0003-reduction/
```

## Practical streaming reference

`miinfer-memory-stream-bench` is a simple contiguous device copy with three
rotating source/destination pairs. It is not a peak HBM claim. Throughput
counts one device read plus one device write.

| Bytes | Median us | Effective GB/s |
| ---: | ---: | ---: |
| 8 MiB | 45.920 | 365.4 |
| 32 MiB | 159.200 | 421.5 |
| 96 MiB | 469.280 | 429.0 |

The copy and GEMV artifacts captured active MI50 telemetry up to 1725 MHz
SCLK, 1000 MHz MCLK, 65 C junction temperature, and 218 W under the 225 W
cap. The sampler is coarse relative to the shortest kernels, so these are
run-state ranges rather than per-kernel counter data.

## Cache regime

The existing controlled HOT comparison is retained separately from STREAMING:

| Regime | HOT us | STREAMING us | HOT delta |
| --- | ---: | ---: | ---: |
| K/V (K) | 70.880 | 70.640 | +0.34% |
| Q/O (Q) | 89.600 | 88.879 | +0.81% |
| G/U (G) | 204.960 | 205.839 | -0.43% |
| D | 231.840 | 231.840 | 0.00% |

Immediate reuse does not materially improve the four regimes. The low K/V
result is not explained by a simple HOT-versus-STREAMING artifact.

## M-scaling diagnostic

`K=4096` was fixed. These are synthetic diagnostics, not canonical model
results; each value is the median of three run medians.

| M | Workgroups | Median us | Effective GB/s |
| ---: | ---: | ---: | ---: |
| 256 | 256 | 20.800 | 101.2 |
| 512 | 512 | 48.800 | 86.1 |
| 1024 | 1024 | 70.400 | 119.3 |
| 2048 | 2048 | 91.040 | 184.4 |
| 4096 | 4096 | 88.160 | 380.8 |
| 8192 | 8192 | 147.520 | 455.1 |
| 12288 | 12288 | 207.200 | 486.0 |

Throughput changes sharply between M=2048 and M=4096, then approaches the
large-grid streaming regime. This supports a small-grid/parallelism limit in
K/V. Q/O is near the transition; G/U is in the saturated regime.

## K-scaling diagnostic

`M=4096`, hence 4,096 output workgroups, was fixed.

| K | Workgroups | Median us | Effective GB/s |
| ---: | ---: | ---: | ---: |
| 1024 | 4096 | 29.760 | 282.2 |
| 2048 | 4096 | 49.280 | 340.7 |
| 4096 | 4096 | 88.320 | 380.1 |
| 8192 | 4096 | 170.560 | 393.6 |
| 12288 | 4096 | 238.080 | 423.0 |

With sufficient output work, time grows approximately with matrix bytes and
bandwidth rises toward the streaming reference. This weakens a
reduction-dominated explanation for the K/V anomaly.

## MIInfer ISA and resources

The gfx906 code object was extracted from the release object with
`clang-offload-bundler` and inspected with ROCm LLVM tools. Baseline metadata:

```text
kernel: fp16_gemv_baseline_kernel
target: amdgcn-amd-amdhsa--gfx906
wavefront: 64
VGPR: 14
SGPR: 20
LDS: 1024 bytes
VGPR/SGPR spills: 0/0
```

The kernel contains 9 `s_barrier` instructions: one after initial LDS partial
publication and eight tree stages. It uses `global_load_ushort`, compiler-
generated `v_fma_mix_f32` packed FP16-pair arithmetic, `ds_write_b32`,
`ds_read*`, and final `global_store_short`; no scratch spill sequence is
present.

The no-reduction diagnostic writes 256 FP32 partials per row, so its stores are
an explicit confounder:

| Regime | Full us | Dot-only us | Difference |
| --- | ---: | ---: | ---: |
| K | 70.400 | 70.400 | 0.0% |
| Q | 87.840 | 124.640 | -41.9% |
| G | 206.880 | 231.360 | -11.8% |
| D | 236.000 | 248.960 | -5.5% |

This diagnostic cannot isolate reduction cost and is not a performance
candidate. No optimization is accepted from it.

## Profiling availability

The host has ROCm 7.1 runtime/compiler and roctracer libraries, but no
`rocprof`, `rocprofv1`, `rocprofv2`, `rocprof-compute`, or `omniperf` command.
Reliable occupancy, VALU, L2/cache, HBM, LDS, and barrier-stall counters are
therefore **UNAVAILABLE**. HIP-event timings, ROCm-SMI telemetry, code-object
metadata, and disassembly were used; counter values were not inferred.

## Pinned llama.cpp decode path

The reference is pinned to `milpster/gfx906-llama-cpp` commit
`6e4ef6c1a553b8f61ad77bba18e9ca05aa677295`; the Qwen3-8B F16 artifact has
SHA256 `c1fd1fc17831ebc0001d81c97a3f78626dd1f977841dec532eef60177abb2a1c`.

A validated one-token `llama-bench` run used `-p 0 -n 1 -r 1 -b 1 -ub 1
-ngl 99 -v`; it found one ROCm gfx906 device and offloaded `37/37` layers.

The actual F16 matrix path is the custom HIP `mmvf` path, not hipBLAS
GEMM-N=1:

```text
ggml_cuda_mul_mat
  -> ggml_cuda_should_use_mmvf
  -> ggml_cuda_mul_mat_vec_f
  -> mul_mat_vec_f<half,...>
  -> HIP kernel mul_mat_vec_f
```

Evidence is mutually consistent: the pinned build compiles
`ggml-cuda/mmvf.cu` with `--offload-arch=gfx906`; the linked
`libggml-hip.so` contains `ggml_cuda_mul_mat_vec_f` and instantiated
`mul_mat_vec_f<__half,...>` symbols; and the F16 model smoke run executes with
all layers on the ROCm gfx906 device. Its source dispatch selects a 256-thread
block for K=4096 under its heuristic. Its activation and output contract is
FP32 around FP16 weights, so it is not semantically identical to EXP-0002's
FP16-input/FP16-output control.

No installed profiler provided named kernel durations or aggregate matrix
share of TG128. That contribution is **UNAVAILABLE**, rather than guessed
from source. This is the reason for the retest status.

## Model-level sanity check

The accepted EXP-0002 medians sum to approximately 963.92 us per layer. Across
36 layers:

```text
963.92 us * 36 = 34.70 ms/token
1 / 34.70 ms = 28.8 tokens/s
```

This is a `SANITY-CHECK LOWER BOUND` / optimistic projection-only ceiling. It
excludes attention, normalization, RoPE, KV work, activation, residuals,
final projection, sampling, and dispatch/fusion differences. It is not a
predicted MIInfer throughput.

## Bottleneck classification

| Regime | Evidence | Classification | Confidence |
| --- | --- | --- | --- |
| K/V | 118.5–118.8 GB/s versus 365 GB/s copy; HOT≈STREAMING; M-sweep rises at M=4096 | grid/parallelism limited, mixed working-set effect | HIGH |
| Q/O | 375–376 GB/s; near M-sweep transition; little cache effect | mixed: streaming plus grid sensitivity | MEDIUM |
| G/U | 489 GB/s at 96 MiB; M-sweep saturates | likely memory-bandwidth limited | HIGH |
| D | 434 GB/s at 96 MiB; K-sweep scales by bytes | likely memory-bandwidth limited, aspect-ratio dependent | MEDIUM |

## Decision

**EXP-0003 RETEST**

The MIInfer regimes and ISA are sufficiently characterized to reject “K/V is
simply HBM-bandwidth limited” and select a next hypothesis. The required
runtime measurement of the pinned llama.cpp matrix-kernel share of TG128 was
not available because the host lacks a compatible profiler.

## Next experiment

The single recommended specialization hypothesis is **different
rows-per-workgroup geometry** for K/V, targeting `M=1024, K=4096`, with Q/O
as a guard shape. The mechanism is to increase independent row work per
dispatched workgroup and reduce the small-grid regime's per-row scheduling and
synchronization exposure while preserving the numerical contract.

Expected effect: move K/V toward the measured 32 MiB streaming regime without
regressing Q/O. Reject if K/V median does not improve by at least 10%, Q/O
regresses by more than 5%, or any correctness metric fails. This candidate is
not implemented here. A compatible ROCm profiler should be exposed before
accepting that follow-up so the external kernel contribution can also be
measured.
