# EXP-0300 — M25-A MIInfer versus mx-llama differential

## Hypothesis

The remaining P512 gap is caused by execution-plan and kernel choices in the
recurrent prefill path, rather than by MI50 hardware limits. Compare the
actual production MIInfer path with the pinned gfx906 `mx-llama.cpp` reference
under the same workload and fixed clocks.

## Baseline and environment

- AMD Instinct MI50 / gfx906, 60 CUs, Wave64
- Qwen3.8-27B-Q4_K_M GGUF (model hash recorded by EXP-0285)
- exact 512-token repeated fox prompt
- context capacity 1024, `-p 512`, `-n 0`, full GPU offload
- MIInfer commit `c6305b3`, row-128 resident-all production path
- mx-llama commit `2e9d29f`
- system ROCm libraries; mx trace reports HIP 7.14.0
- manual DPM: SCLK 1606 MHz, MCLK 1000 MHz

MIInfer used the qualified switches:

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
```

Both runs were sampled continuously with `rocm-smi`. The mx run had 83/83
samples at 1606/1000 MHz; MIInfer had 223/223 samples at 1606/1000 MHz.

## End-to-end P512

| runtime | latency | throughput | allocation |
|---|---:|---:|---:|
| mx-llama | 2317.872 ms | 220.892 tok/s | not exported by benchmark |
| MIInfer | 5050.180 ms | 101.383 tok/s | 22,801,772,884 B |

MIInfer is therefore `2.179x` slower, with a `2732.308 ms` latency gap. These
are valid clock-qualified measurements, but the mx measurement was collected
under ROCProfiler tracing while the MIInfer throughput run was unprofiled;
the profiler overhead is not treated as a MIInfer-versus-mx conclusion.

## mx-llama kernel trace

`rocprofv3 --runtime-trace --kernel-trace --memory-copy-trace` captured 4,884
dispatch rows. The named compute kernels dominate the trace as follows (the
trace includes the process startup sequence; times are summed dispatch time,
not wall latency):

| kernel family | dispatches | summed GPU ms |
|---|---:|---:|
| `mmq_gemm_repacked` Q4_K | 576 | 3084.396 |
| `mmq_gemm_repacked` Q6_K | 128 | 664.752 |
| `mmq_gemm_repacked` Q5_K | 96 | 274.814 |
| `gated_delta_net_chunked_cuda` | 96 | 189.115 |
| rocBLAS GEMM (`Cijk_...`) | 192 | 120.640 |
| `flash_attn_tile` | 32 | 28.315 |
| `quantize_mmq_q8_1` | 800 | 35.149 |
| `unary_gated_op_kernel` | 256 | 28.035 |
| `ssm_conv_long_token_f32` | 96 | 18.723 |

The Q4/Q5/Q6 mapping is from the pinned ggml enum (`Q4_K=12`, `Q5_K=13`,
`Q6_K=14`). The trace exposes workgroup/grid sizes and static LDS/VGPR
metadata, but this capture did not enable PMC counters, so HBM/L2 traffic,
active-wave occupancy and CU utilization are not claimed.

## MIInfer whole-B512 profile

The existing MIInfer event profiler was run with the same switches and clocks.
Its instrumentation adds wall overhead, so its `5488.79 ms` wall value is not a
throughput qualification. The whole-wide recurrent attribution is valid:

| recurrent family | total GPU ms | ms/layer (48 layers) |
|---|---:|---:|
| GDN core | 813.842 | 16.955 |
| SSM output + residual/post-norm | 291.729 | 6.078 |
| FFN Gate/Up + SwiGLU | 1611.800 | 33.579 |
| FFN Down + residual | 802.021 | 16.709 |
| recurrent deferred tail | 3520.380 | 73.341 |

Attention deferred tails total `936.410 ms` across 16 layers. Repacked-weight
uploads remained zero and the resident allocation was unchanged at
22,801,772,884 B.

The MIInfer profile also reports logical wide-prefill launches per layer:
recurrent `qkv`, `gate`, `beta_alpha`, `ssm_out`, `ffn_gate`, `ffn_up`, and
`ffn_down` (48 each), plus attention `qk`, `v`, `q_post`, `k_post`,
`attention`, `o`, `ffn_gate`, `ffn_up`, and `ffn_down` (16 each). These are
runtime operation counters, not directly comparable to mx kernel-dispatch
rows because several MIInfer operations expand into multiple HIP kernels.

## Correctness

The MIInfer P512 run completed with finite output and no validation failure.
The resident-all canonical validation and layer tolerances remain those
qualified by EXP-0299. No implementation change was made in this experiment.

## Interpretation

The differential is decisive at the architectural level: MIInfer is roughly
2.18x slower at P512 even with resident weights and identical manual clocks.
The mx trace shows a mature Q4/Q5/Q6 repacked-MMQ pipeline with activation
quantization reuse and specialized recurrent kernels. MIInfer's measured
recurrent tail is still 3.520 s, so the next target is the recurrent execution
contract and its materialization/launch structure, not another isolated
attention projection.

Per-dispatch MIInfer ROCProfiler data is still missing. Starting the server
with `ROCP_TOOL_ATTACH=1` did not expose the required `rocp-bg-attach` thread,
and `rocprofv3 --pid` therefore failed before collection. A full cold-process
MIInfer kernel trace was rejected because profiling model initialization
introduced multi-minute stalls. Do not present the current aggregate table as
a complete kernel-for-kernel accounting.

## Decision

**KEEP the differential and proceed to M25-B/C/D. RETEST only the missing
MIInfer per-dispatch trace if an attach-enabled build or a small recurrent
harness becomes available.** Do not spend more time on QK/V/O projection
micro-tuning before the recurrent prefill contract is changed and measured.

## Follow-up

1. Do not reimplement Gate/Up Q8 reuse: it is already the production default
   and was qualified by EXP-0029.
2. Do not retry the existing FFN norm→Q8 or SwiGLU→Q8 mappings: EXP-0242 and
   EXP-0243 rejected them at end-to-end scale. Any M25 fusion must use a new
   mapping and a new A/B hypothesis.
3. Compare the proven B64 GDN topology and gfx906 workgroup geometry with the
   pinned mx implementation.
4. Gate every change by absolute milliseconds removed from the qualified
   4.8–5.0 s P512 baseline.
