# EXP-0361 — Wave64 attention architecture analysis

## Question

Why can the mx-llama.cpp gfx906 attention path outperform MIInfer at long
prefill, and which architectural differences are worth isolating before any
new MIInfer kernel is written?

## Scope

Source archaeology only. No MIInfer production code was changed and EXP-0360
remains rejected.

## MIInfer control

Source: `gfx906/kernels/qwen3_primitives.hip`.

| Property | Current MIInfer prefill path |
| --- | --- |
| Workgroup | 64 threads / one Wave64 |
| Grid | one workgroup per `(query token, query head)` |
| Q heads / KV heads | 24 / 4; KV head selected as `head / 6` |
| Query-token blocking | none |
| K/V type | production F16 or F32 KV; qualified path is F16 |
| KV traversal | each wave scans its causal prefix serially |
| Softmax | per-wave online max/sum/accumulator |
| Inter-wave barriers | none |
| GQA sharing | implicit cache opportunity only; no explicit sharing |

## mx-llama.cpp path

Repository: `/home/fedora-workstation/Development/mx-llama.cpp` at commit
`2e9d29fe`.

Relevant sources:

* `ggml/src/ggml-cuda/fattn.cu`: best-kernel selection and GQA ratio logic;
* `ggml/src/ggml-cuda/fattn-tile.cuh`: AMD tile configurations and dispatch;
* `ggml/src/ggml-cuda/fattn-common.cuh`: grid geometry and tiled execution;
* `ggml/src/ggml-cuda/fattn-vec.cuh`: alternative 128-thread vector path.

For Qwen3’s shape (head dimension 256, GQA ratio 6), the source-selected
long-prefill tile path is:

| Property | mx source-derived configuration |
| --- | --- |
| Kernel family | generic `flash_attn_tile<256,256,...>` |
| GQA grouping | `ncols2 = 2` query heads per KV head tile |
| Long-prefill query blocking | `ncols1 = 16` tokens; 32 query rows/block |
| Workgroup | 256 logical threads; HIP source uses software warp size 32, i.e. 4 hardware Wave64s on gfx906 |
| AMD tile config for 32 columns | `nbatch_fa = 32`, `nbatch_K = 128`, occupancy target 2 |
| K/V staging | tile kernel stages K/V into shared memory and reuses it across query rows within the block |
| KV traversal | blocks advance over `nbatch_fa=32` KV rows; `grid.y` can split the KV scan |
| GQA output grid | `grid.z = ceil(6/2) * 4 = 12` per sequence |
| K/V types | the backend supports Q8_0 K and F16 V; tile launch requests F16 K/V staging, converting non-F16 inputs into temporary F16 buffers |

The `ncols2=2` and `ncols1=16` values are source-derived for long prefill,
not a benchmark claim. Exact runtime selection should be confirmed with mx’s
verbose dispatch logging or a trace on the same model and prompt.

## Architectural difference

The important difference is not simply “FlashAttention” or six-way GQA
cooperation:

```text
MIInfer:  one token × one Q head × one Wave64; serial causal scan
mx:      16 query tokens × 2 Q heads per KV head; tiled K/V scan
```

mx obtains reuse across neighboring query tokens and a small number of GQA
heads while keeping the workgroup bounded at 256 threads. It does not use the
EXP-0360 six-wave `(one token, six Q heads)` workgroup. Its tile kernel also
uses compile-time AMD-specific `nbatch_fa`/`nbatch_K` choices and can split
the KV traversal across `grid.y`, followed by online-softmax fixup.

## MIInfer compiler resource snapshot

ROCm Clang 20.0.0, `--offload-arch=gfx906`, compiled with
`-mllvm -amdgpu-dump-hsa-metadata` from the production compile command:

| Kernel | VGPR | SGPR | VGPR spills | SGPR spills | Wavefront |
| --- | ---: | ---: | ---: | ---: | ---: |
| MIInfer control F16 | 32 | 25 | 0 | 0 | 64 |
| EXP-0360 candidate F16 | 35 | 31 | 0 | 0 | 64 |

The candidate’s dynamic LDS allocation is 32768 bytes at launch; the control
has no dynamic LDS allocation. These compiler facts explain some resource
pressure, but do not by themselves explain the full 1.77x P4K slowdown.

## Evidence status

Confirmed from source:

1. mx has a concrete long-prefill query-token tile path;
2. its Qwen-compatible GQA grouping is `ncols2=2`, not six-wave grouping;
3. its gfx906/AMD tile configuration for 32 columns is `256 threads`,
   `nbatch_fa=32`, `nbatch_K=128`, occupancy target 2;
4. MIInfer control has no query-token blocking.

Not yet confirmed:

* exact internal tile-kernel dispatch geometry in the verbose benchmark output;
* compiled VGPR/SGPR/scratch counts for mx and MIInfer;
* whether mx’s `grid.y` uses multiple KV splits at P4K/P8K;
* whether Q8 K conversion cost is amortized or dominates.

## mx runtime confirmation

Command used:

```text
mx-llama-build/bin/llama-bench \
  -m Qwen3.8-27B-Q4_K_M.gguf -pg 4096,0 -b 4096 -ub 4096 \
  -ctk q8_0 -ctv f16 -fa on -ngl 99 -sm none -r 1 -v
```

The completed run reported:

* MI60/MI50, gfx906, Wave Size 64;
* flash attention enabled;
* KV cache K=`q8_0`, V=`f16`;
* Qwen3.8-27B: 24 Q heads, 4 KV heads, head dimension 256;
* `pp4096`: `234.21` tokens/s for the mx benchmark run.

This confirms the cache types and runtime flash-attention path, but not the
internal block geometry. The latter remains source-derived until a kernel
trace or more selective verbose dispatch log is captured.

## Decision

Do not write a new MIInfer attention kernel yet. The first justified follow-up
is a matched runtime trace/resource comparison of the existing MIInfer control
and mx’s selected tile configuration. The first portable hypothesis, if that
comparison confirms the source path, is bounded query-token blocking with
independent Wave64/subwave work—not explicit six-wave GQA cooperation.
