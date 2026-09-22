# EXP-0362 — Source-shaped query-tiled GQA2 prefill

## Hypothesis

The mx-shaped geometry will amortize LDS synchronization across enough query
rows to make bounded multi-wave cooperation effective on gfx906:

```text
16 query tokens × 2 Q heads per KV head = 32 query rows/workgroup
256 threads = 4 physical Wave64s
32 K/V rows per staged tile
128-wide Q·K dimension chunks (source-shaped parameter)
F16 K/V, online softmax, causal masking
```

## Baseline

The existing MIInfer control is one Wave64 per token/query-head pair. EXP-0360
was the rejected six-wave, one-token/six-head design.

## Candidate

An opt-in isolated kernel was added beside the existing paths. Each 256-thread
workgroup stages a 32×256 K/V tile in 32 KiB LDS. Each Wave64 processes eight
query rows, retaining eight online-softmax accumulators while the staged tile is
shared by all four waves.

## Environment

* AMD MI50/MI60, gfx906, Wave64
* ROCm HIP 7.1.52802-9999
* Qwen3.8 shape: 24 Q heads, 4 KV heads, head dimension 256
* F16 K/V

## Correctness

Exact against the control for all tested lengths:

```text
max_abs_error = 0
```

## Results

| Tokens | Control ms | EXP-0360 ms | EXP-0362 ms | Control / EXP-0362 |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 2.590 | 4.550 | 13.888 | 0.186× |
| 2048 | 41.907 | 77.615 | 189.183 | 0.222× |
| 4096 | 175.377 | 312.660 | 716.948 | 0.245× |
| 8192 | 754.422 | 1214.910 | 2848.240 | 0.265× |
| 16384 | 3292.330 | 4859.070 | 11185.300 | 0.294× |

The source-shaped geometry is substantially slower than both the control and
EXP-0360. Correctness is not the issue.

## Interpretation

The result rejects the literal mx geometry as a direct MIInfer transplant.
Synchronization amortization alone does not compensate for maintaining eight
independent online-softmax states per Wave64 and repeatedly traversing the
shared tile for those rows. The source-shaped workgroup also increases register
and instruction pressure substantially relative to the one-row control.

This does not prove that query-token tiling is impossible, nor does it prove
that mx's implementation has the same execution organization as this kernel.
It does show that copying the visible BQ/GQA/LDS geometry without matching its
QK/vectorization and resource schedule is not a viable optimization.

The compiled metadata confirms the cause:

| Kernel | VGPR | VGPR spills | SGPR | SGPR spills | Private segment | Dynamic LDS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Control | 32 | 0 | 25 | 0 | 0 | 0 KiB |
| EXP-0360 | 35 | 0 | 31 | 0 | 0 | 32 KiB |
| EXP-0362 | 64 | 137 | 52 | 0 | 204 bytes | 32 KiB |

For comparison, the built mx gfx906 fatbin contains the selected
`flash_attn_tile<256,256,16,2>` specialization with 256 threads, 27,136 bytes
of fixed LDS, 97 VGPRs, 46 SGPRs, and zero VGPR spills. Therefore the geometry
itself is not sufficient to reproduce mx: mx sustains a much larger register
footprint without spilling, implying materially different vectorization and
state scheduling.

The eight per-wave accumulator/state sets caused register exhaustion and
spilling. This is sufficient attribution for the rejection; no profiler
installation is needed. Any later multi-row candidate must keep state bounded
enough to avoid spills.

## Decision

REJECT — do not enable this candidate and do not merge it into production.

## Follow-up

Do not enable this candidate. A direct BQ16/GQA2 transplant has failed on the
measured resource constraint. Any future attempt would need to reproduce the
mx implementation’s spill-free state schedule, not merely its launch geometry.
