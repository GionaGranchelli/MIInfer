# EXP-0002 — FP16 GEMV Baseline on Qwen3-8B Shapes

**Status:** KEEP
**Milestone:** M1 — Kernel Laboratory
**Author:**
**Date:** 2026-08-29
**Baseline commit:** `b7dfbddce33f2dfd6d006bfe198a02a6f27dbf7f`
**Candidate commit:** Not applicable — baseline measurement only

---

# 1. Question

What is the correctness and timing baseline for a straightforward FP16 GEMV
implementation on the exact dense Qwen3-8B projection shapes used by the first
MIInfer control model?

# 2. Hypothesis

The seven projection shapes expose different behavior across the 8 MiB,
32 MiB, and 96 MiB matrix-size regimes. The result should identify whether the
first specialization experiment should target bandwidth, launch/shape
behavior, or arithmetic throughput.

# 3. Motivation

M0 freezes the external control model and the benchmark laboratory. M1 needs
actual model dimensions and a correctness-trusted project-owned reference
kernel before any gfx906-specific instruction or layout work is proposed.

# 4. Scope and exclusions

This experiment establishes a baseline only. It does not implement or evaluate
`v_dot2`, `v_dot4`, DPP reduction, `ds_swizzle`, logical Wave32 partitioning,
weight repacking, prefetch, manual ISA, shape-specific tables, autotuning,
fusion, or HIP graphs.

# 5. Baseline implementation

The project-owned implementation is a deliberately straightforward HIP kernel:

* row-major `A[M,K]`, FP16 weights and FP16 input vector;
* one 256-thread workgroup per output row;
* strided FP16 loads and FP32 partial accumulation;
* shared-memory tree reduction of 256 FP32 partials;
* FP16 rounded output;
* all allocation, copies, handle creation, and synchronization outside the
  timed operation.

The deterministic data generator uses seed `0x4D493050` (`1296642128`),
`std::mt19937`, and FP16-rounded values uniformly distributed in `[-0.25,
0.25]`. The CPU oracle multiplies the stored FP16 values after converting them
to FP32 and accumulates sequentially in FP32.

Correctness passes when each finite element satisfies the combined tolerance
condition `absolute_error <= 0.05` or `relative_error <= 0.01`; NaN and Inf
always fail. Relative error uses `max(abs(reference), 1e-6)` as its denominator.

# 6. rocBLAS comparison

The Fedora ROCm 7.1 headers do not provide an FP16 `hipblasHgemv`/`rocblas_hgemv`
path with the required FP16-input, FP16-weight, FP32-accumulate semantics.
`hipblasSgemv` would change the operand contract, so it is not a valid
comparison. The strongest valid installed-library path tested here is
`hipblasGemmEx` with:

```text
A: FP16, B: FP16, C: FP16
compute type: HIPBLAS_COMPUTE_32F
alpha: FP32 1.0, beta: FP32 0.0
operation: HIPBLAS_OP_T / HIPBLAS_OP_N
```

MIInfer's row-major `M x K` bytes are presented as a column-major `K x M`
matrix and transposed by hipBLAS. The timed call is therefore an equivalent
`M x K` by `K x 1` operation without a timed transpose or repack.

# 7. Hardware and software

```text
GPU: AMD Instinct MI60 / MI50
PCI ID: 1002:66A1
Architecture: gfx906 / Vega20
Wavefront: 64
Compute units: 60
VRAM: 34342961152 bytes
Kernel: Linux 7.1.10-200.fc44.x86_64
Distribution: Fedora Linux 44 Workstation
CPU: Intel Xeon E5-2680 v3 @ 2.50 GHz
HIP: 7.1.52802-9999
HIP compiler: Clang 20.0.0.rocm7.1.1
ROCm runtime: rocm-runtime-7.1.1-6.fc44
rocBLAS: rocblas-7.1.1-7.fc44 / rocblas-devel-7.1.1-7.fc44
hipBLAS: hipblas-7.1.0-6.fc44 / hipblas-devel-7.1.0-6.fc44
rocminfo: 7.1.0-4.fc44
```

The host was in the M0-approved state: gfx802/Tonga was unbound from
`amdgpu`/KFD, `rocminfo` exposed gfx906, MIInfer device validation passed, and
no Ollama, llama-server, or LM Studio process was running. The release preset
compiled for gfx906. The five canonical runs all report commit
`b7dfbddce33f`, `git_dirty=false`.

# 8. Test matrix — real model shapes

The convention is `M x K`: the input vector has length `K` and the output has
length `M`.

| ID | Projection | M | K | Matrix bytes |
| --- | --- | ---: | ---: | ---: |
| Q | Q projection | 4096 | 4096 | 32 MiB |
| K | K projection | 1024 | 4096 | 8 MiB |
| V | V projection | 1024 | 4096 | 8 MiB |
| O | Output projection | 4096 | 4096 | 32 MiB |
| G | FFN gate projection | 12288 | 4096 | 96 MiB |
| U | FFN up projection | 12288 | 4096 | 96 MiB |
| D | FFN down projection | 4096 | 12288 | 96 MiB |

# 9. Test infrastructure

Added targets:

```text
miinfer-fp16-gemv-test
miinfer-fp16-gemv-bench
```

The correctness test covers two small non-multiple shapes (`7 x 13` and
`257 x 37`) plus all seven real shapes on the physical MI50. Debug and Release
CTest both include the GPU-required correctness test. The benchmark emits
JSONL with every individual HIP-event sample, correctness metrics, effective
bandwidth, build metadata, and project-kernel resource metadata.

# 10. Cache regime and benchmark command

The canonical result is `STREAMING`, implemented by rotating among three
identical device weight buffers. The copies are outside the timed region. This
avoids treating repeated immediate reuse of one matrix as the inference result.
The operation-only timed region contains the project kernel or the single
hipBLAS call.

Canonical command, repeated five times independently:

```text
scripts/run-bench.sh ./build/mi50-release/miinfer-fp16-gemv-bench \
  --shape all --implementation all --cache-regime streaming \
  --warmup 10 --iterations 300
```

The fixed executable order is `miinfer-baseline` followed by `rocblas-gemm`
for each shape. Each run contains 14 result records and 300 raw timed samples
per record. The runner retains `environment-before.json`,
`telemetry.jsonl`, `result.json`, and `environment-after.json`.

Canonical run directories:

```text
bench/results/20260829T210120Z-52467/
bench/results/20260829T210131Z-52849/
bench/results/20260829T210142Z-53252/
bench/results/20260829T210152Z-53637/
bench/results/20260829T210203Z-54018/
```

A separate, non-canonical HOT comparison was also retained at:

```text
bench/results/20260829T210403Z-54877/
```

It used the same command with `--cache-regime hot`; its results are not mixed
with the canonical streaming aggregate.

# 11. Correctness results

All 70 canonical shape/implementation records passed. No NaN or Inf was
observed. Representative maximum absolute error by shape was:

| Shape | MIInfer max abs | rocBLAS max abs | MIInfer mean abs | rocBLAS mean abs | Cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q/O | 0.001893520 | 0.001893520 | 0.000188798 | 0.000188803 | 0.999999978 |
| K/V | 0.001878262 | 0.001878262 | 0.000187762 | 0.000187773 | 0.999999978 |
| G/U | 0.001787663 | 0.001787663 | 0.000188630 | 0.000188636 | 0.999999978 |
| D | 0.001943111 | 0.001943111 | 0.000321585 | 0.000321606 | 0.999999978 |

Maximum relative error is not used alone for acceptance because values close to
zero amplify it; the combined absolute/relative rule is explicit above.

# 12. Canonical performance results

Values below are the median of the five per-run medians. Effective bandwidth is
logical FP16 matrix + vector input + vector output bytes divided by that median;
it is not a profiler measurement of HBM traffic.

| Shape | M | K | Implementation | Median us | Effective GB/s | Cross-run SD us | Correctness |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| Q | 4096 | 4096 | MIInfer baseline | 89.520000 | 375.009 | 0.394 | PASS |
| Q | 4096 | 4096 | rocBLAS GEMM-N=1 | 318.559498 | 105.383 | 0.697 | PASS |
| K | 1024 | 4096 | MIInfer baseline | 70.720002 | 118.762 | 0.088 | PASS |
| K | 1024 | 4096 | rocBLAS GEMM-N=1 | 216.000006 | 38.884 | 0.812 | PASS |
| V | 1024 | 4096 | MIInfer baseline | 70.880003 | 118.494 | 0.465 | PASS |
| V | 1024 | 4096 | rocBLAS GEMM-N=1 | 216.479503 | 38.797 | 1.189 | PASS |
| O | 4096 | 4096 | MIInfer baseline | 89.279499 | 376.019 | 0.206 | PASS |
| O | 4096 | 4096 | rocBLAS GEMM-N=1 | 320.959494 | 104.595 | 1.074 | PASS |
| G | 12288 | 4096 | MIInfer baseline | 205.839500 | 489.197 | 0.221 | PASS |
| G | 12288 | 4096 | rocBLAS GEMM-N=1 | 364.720002 | 276.091 | 0.742 | PASS |
| U | 12288 | 4096 | MIInfer baseline | 205.760002 | 489.386 | 0.157 | PASS |
| U | 12288 | 4096 | rocBLAS GEMM-N=1 | 365.039513 | 275.850 | 1.510 | PASS |
| D | 4096 | 12288 | MIInfer baseline | 231.919996 | 434.184 | 1.204 | PASS |
| D | 4096 | 12288 | rocBLAS GEMM-N=1 | 941.278994 | 106.978 | 1.129 | PASS |

The project-owned baseline is the fastest correctness-valid implementation in
all seven shapes under this canonical regime. This is a kernel-level result,
not an end-to-end inference claim.

# 13. Hardware state and contamination

Across the 110 canonical telemetry records, the observed ranges were:

```text
all records:   SCLK 925–1725 MHz, MCLK 350–1000 MHz,
               junction 34–49 C, power 19–165 W
active records (SCLK >= 1000 MHz):
               SCLK 1725 MHz, MCLK 800–1000 MHz,
               junction 38–49 C, power 72–165 W
```

Each canonical run retained 22 telemetry records. The MI50 reached the
expected 1725 MHz SCLK and up to 1000 MHz MCLK during active work. No competing
workload was present, no accepted run was marked contaminated, and the power
cap remained 225 W.

# 14. Kernel resources

Runtime HIP attributes were identical for all seven project-owned shapes:

```text
workgroup: 256 threads
VGPRs: 14
LDS/shared memory: 1024 bytes
local memory: 0 bytes
max threads per block: 1024
SGPR count: not exposed by hipFuncAttributes
```

The Release object contains the `amdgcn-amd-amdhsa--gfx906` code object and the
baseline kernel symbol. No separate disassembly tool was available in the host
environment; no ISA optimization or spill investigation was performed in this
baseline experiment.

# 15. Interpretation

The 96 MiB G/U/D shapes reach the highest effective bandwidth and are most
consistent with a bandwidth-dominated baseline. Q/O are also substantial
matrix streams, with their lower logical size making cache and launch behavior
more relevant. K/V have only 8 MiB of weights and the smallest output grid;
their low effective bandwidth makes launch/shape behavior and cache regime
important. These are working classifications, not profiler-confirmed HBM
traffic diagnoses.

# 16. Decision

```text
EXP-0002 KEEP
```

All seven real shapes are correctness-valid, the project-owned baseline and
strongest valid installed-library comparison are reproducible, raw samples and
hardware telemetry are retained, and cross-run variation is small. This result
freezes the FP16 GEMV baseline for future isolated experiments.

# 17. Follow-up recommendation

Recommend one smallest next experiment only: establish a quantized GEMV
baseline for the same seven shapes, or, if the project chooses to stay with
FP16, profile the K/V and 96 MiB regimes before selecting the first
gfx906-specific candidate. No follow-up optimization is implemented here.
