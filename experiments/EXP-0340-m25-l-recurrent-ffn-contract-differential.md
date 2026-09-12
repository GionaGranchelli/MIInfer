# EXP-0340 — M25-L recurrent FFN contract differential

**Status:** RETEST; measurement-only, no optimization
**Milestone:** M25-L
**Date:** 2026-09-12
**MIInfer source:** `dc94b4d`
**Reference:** `mx-llama.cpp@2e9d29fe736969160f17476ec6f0a6298cee6966`

## Question

Is the current stretch gap large enough, and sufficiently localized to the
recurrent FFN, to justify writing another candidate?

## Environment

- AMD Instinct MI50 / gfx906 / Wave64
- Qwen3.8-27B-Q4_K_M; SHA-256
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- six interleaved fresh-process pairs: mx first, then MIInfer
- fixed SCLK/MCLK `1606/1000 MHz`
- no profiler for the primary comparison
- continuous 250 ms telemetry: `1413` samples, all at `1606/1000 MHz`,
  maximum junction temperature `59 C`

MIInfer used the hermetic `MIINFER_PRESET=m25_hi_qualified` vector and the
exact repeated-fox prompt. The pinned `llama-bench` executable accepts only a
token count, not prompt text, so its P512 samples use the benchmark's synthetic
prompt tokens. An exact-text `llama-completion` run was measured separately at
`2599.15 ms` / `196.99 tok/s`, but it uses a different execution path (`graphs
reused=0`) and is excluded from the oracle comparison.

## Primary benchmark

Commands, with one fresh process per sample:

```text
mx-llama-build/bin/llama-bench -m MODEL -ngl 99 -fa on -ctk q8_0 -ctv f16 \
  -b 2048 -ub 2048 -p 512 -n 0 -r 1 -o json

build/mi50-release/miinfer run MODEL --prompt EXACT_512_TOKEN_PROMPT \
  --max-tokens 0 --no-stream
```

Raw P512 latency samples were:

| process | mx-llama.cpp (ms) | MIInfer H/I (ms) |
|---:|---:|---:|
| 1 | 2308.736 | 2618.82 |
| 2 | 2307.979 | 2606.95 |
| 3 | 2308.688 | 2508.44 |
| 4 | 2304.878 | 2495.59 |
| 5 | 2307.290 | 2473.30 |
| 6 | 2306.063 | 2475.92 |
| **median** | **2307.634** | **2474.610** |

The fresh screen therefore reports:

```text
mx-llama.cpp: 221.872 tok/s
MIInfer H/I:  206.901 tok/s
latency gap:  166.976 ms (7.236%)
```

MIInfer processed exactly 512 tokens in all six runs, with fixed allocation
`21,993,242,964 B` and `11,484,004,352 B` free immediately after setup.
The H/I continuation and repeat-P512 correctness gates remain those qualified
by EXP-0339; this screen itself requested zero generated tokens.

## Current MIInfer recurrent-layer measurement

The existing diagnostic profiler was run on the exact prompt and current H/I
vector. It is diagnostic only because event instrumentation changes wall
time; the reported wide-tail events cover the complete B512 family boundary.
The profiled run reported `2433.72 ms` of CLI wall time and `210.38 tok/s`.
Those numbers are diagnostic only because the profiler changes execution
timing.

For representative recurrent layer 60 at the final B512 chunk:

| stage | whole-B512 GPU ms |
|---|---:|
| GDN core | 3.57872 |
| SSM output + residual | 2.95680 |
| FFN Gate/Up + SwiGLU | 13.8275 |
| FFN Down + residual | 8.44191 |
| **FFN total** | **22.2694** |

Across all 48 recurrent layers the same profiler reported:

| stage | whole-B512 GPU ms |
|---|---:|
| GDN core | 185.226 |
| SSM output + residual | 141.210 |
| FFN Gate/Up + SwiGLU | 666.870 |
| FFN Down + residual | 381.717 |
| **FFN total** | **1048.587** |

## Current oracle kernel trace

A current no-candidate oracle trace was captured with ROCProfiler 1.3.1 around
the pinned `llama-bench` P512 shape. It reported `2311.302 ms` for that single
profiled run. The dominant source-level families were:

| kernel family | dispatches | summed GPU ms |
|---|---:|---:|
| `mmq_gemm_repacked` Q4_K | 576 | 3079.979 |
| `mmq_gemm_repacked` Q6_K | 128 | 664.536 |
| `mmq_gemm_repacked` Q5_K | 96 | 272.755 |
| `gated_delta_net_chunked_cuda` | 96 | 189.175 |
| rocBLAS GEMM | 192 | 120.619 |
| `quantize_mmq_q8_1` | 800 | 35.076 |
| `unary_gated_op_kernel` | 224 | 25.776 |

The summed dispatch time exceeds wall time because the trace contains
overlapping kernel work. Kernel names and dimensions do not carry tensor
identity, so this is not yet a valid recurrent-FFN stage table. In particular,
the trace cannot prove how much of the `166.976 ms` current end-to-end gap is
FFN versus GDN, attention, or materialization.

## Interpretation

The current requalification narrows the historical `182.748 ms` gap to
`166.976 ms` under this session, but the primary comparison still has the
oracle prompt-content limitation. The MIInfer profile confirms that FFN is a
large repeated local cost, not that it owns the differential. The static graph
rejection remains negative evidence against generic host-launch overhead.

The requested full contract table is therefore not complete yet:

| contract stage | oracle | MIInfer | decision |
|---|---:|---:|---|
| post-attention norm through Q8 | not isolated | measured in existing profile | keep measuring |
| Gate/Up MMQ | mixed kernel trace | `666.870 ms` with SwiGLU | unresolved |
| SwiGLU | mixed kernel trace | included above | unresolved |
| Down MMQ | mixed kernel trace | `381.717 ms` with residual | unresolved |
| whole recurrent FFN span | not isolated | `1048.587 ms` | unresolved |

## Decision

**RETEST.** Do not write an M25-L optimization yet. The next measurement must
provide matching recurrent-FFN boundaries on both runtimes, ideally one real
recurrent layer at B512 with captured activation/state and an outer event pair;
the decomposition trace can then be used only as a secondary diagnostic.

## Follow-up

1. Preserve the current MIInfer stage events and add no candidate selector.
2. Obtain oracle tensor/stage identity for one recurrent layer or an equivalent
   marker-based trace.
3. Only if the oracle-to-MIInfer FFN delta is about `3.5 ms/layer` or more,
   evaluate a complete contract port. Otherwise redirect the stretch work to
   the measured differential family.

## Re-evaluation — 2026-09-12

The pinned oracle was temporarily instrumented at source level, with graphs
disabled so HIP events were not recorded during graph capture. The trace was
labelled by tensor name for recurrent layer 60 and compared with MIInfer's
existing layer-60 B512 profile:

| contract stage | mx oracle (ms) | MIInfer (ms) | observation |
|---|---:|---:|---|
| QKV projection | 3.958 | 4.643 | concrete Q6 MMQ differential |
| GDN core | 1.924 | 3.579 | source-labelled event boundaries differ; retest |
| FFN Gate/Up + SwiGLU | about 14.5 | 13.828 | not slower in MIInfer |
| FFN Down + residual | about 7.6 | 8.442 | near-tie at this boundary |
| whole recurrent FFN | 22.117 | 22.269 | only about `0.15 ms/layer` |

The oracle trace is not a qualification: it uses synthetic benchmark tokens,
disabled graphs, and per-node diagnostic events. It is nevertheless sufficient
to reject the original FFN localization hypothesis. A faithful pinned Q6 MMQ
selector was added only for recurrent prefill QKV as
`MIINFER_MX_PINNED_QKV=1`. At the exact layer-60 QKV B512 shape, the isolated
MMQ median moved from `4516.955 us` to `4032.155 us`, with maximum absolute
contract error `3.6e-7`. Three fresh-process end-to-end pairs measured:

```text
control:    2486.07, 2500.70, 2761.63 ms  (median 2500.70 ms)
pinned QKV: 2455.94, 2516.61, 2492.25 ms  (median 2492.25 ms)
```

The `2761.63 ms` control is retained as an outlier rather than silently
discarded. The `0.34%` median improvement is below qualification confidence,
so the selector remains opt-in and the qualified preset explicitly sets it to
`0`. A B512 GDN standalone retest of explicit loop unrolling changed
`2834.876 us` to `2832.636 us` (`0.08%`) with unchanged numerical parity and
was rejected. A direct 32-bit HIP shuffle mask copied from the oracle does not
compile under the current HIP headers, which require a 64-bit mask.

## Re-evaluation decision

**RETEST / REDIRECT.** The FFN candidate is rejected as unjustified. Keep the
QKV pinned port available for future interleaved qualification, and make the
next measurement a like-for-like QKV/GDN execution-contract differential.
