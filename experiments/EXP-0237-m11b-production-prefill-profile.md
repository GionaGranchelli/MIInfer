# EXP-0237 — M11-B production layer-major prefill profile

## Question

Which production layer-major stages must a new causal chunk schedule improve
to reach the 100 PP tok/s gate?

## Method

Added a disabled-by-default `MIINFER_PREFILL_PROFILE=1` mode to the production
CLI. It uses HIP events around each layer's three layer-major regions:

1. `prepare`: normalization, activation quantization, and batched input
   projections;
2. `ordered`: position-ordered recurrent/attention work and state/KV updates;
3. `tail`: deferred SSM/O projection, residual/norm, FFN, and output stages.

Embedding is timed separately. The profiler synchronizes after each layer and
therefore adds measurement overhead; its wall time is not used as the
qualified throughput result.

## Environment

- GPU: AMD Instinct MI50, gfx906
- Model: `Qwen3.8-27B-Q4_K_M.gguf`
- Build: `mi50-release`
- ROCm: 6.4.0 / LLVM 20
- Prompt: repeated `hello` tokens
- `MIINFER_PREFILL_LAYER_MAJOR=1`
- `MIINFER_PREFILL_PROFILE=1`
- generation disabled with `--max-tokens 0`

## Results

The current non-profiled production baseline at this HEAD was:

| Workload | Prefill | PP tok/s |
| --- | ---: | ---: |
| P128 | 2681.24 ms | 47.74 |
| P512 | 11163.67 ms | 45.86 |

A P128/TG64 run measured `32.47 tok/s` average decode, preserving the
M11-A short-context decode range. These are single qualification checks; the
repeated P513 result in EXP-0235 remains the performance claim.

Profiled GPU-region totals, in milliseconds:

| Prompt | Family | Prepare | Ordered | Tail | Layer total |
| ---: | --- | ---: | ---: | ---: | ---: |
| P128 | 48 recurrent | 383.9 | 203.0 | 1497.4 | 2084.2 |
| P128 | 16 attention | 114.6 | 88.4 | 420.1 | 623.1 |
| P512 | 48 recurrent | 1524.9 | 815.4 | 6074.4 | 8414.7 |
| P512 | 16 attention | 461.9 | 731.6 | 1682.3 | 2875.8 |

The P512 profiled wall time was approximately `11.35 s`; the layer regions
accounted for `11.29 s`, with about `9 ms` of measured embedding and about
`70 ms` of remaining wall time plus profiling/synchronization effects. The
qualified non-profiled P513 result remains approximately `46.2 PP tok/s`.

At P512, recurrent layers account for `8.41 s` of measured layer work and
attention layers for `2.88 s`. Across the recurrent family, the deferred tail
is the largest bucket (`6.07 s`), followed by the prepared projections
(`1.52 s`). The ordered recurrent dependency is only `0.82 s`. Attention's
ordered region grows with context (`0.73 s` at P512 versus `0.09 s` at P128),
but remains smaller than recurrent tail work.

## Amdahl interpretation

The current sequential recurrent state update is not the dominant floor. An
ideal removal of the ordered recurrent region would reduce P512 only from
about `11.3 s` to `10.5 s`, far short of the `5.12 s` 100 PP tok/s target.
The next design must increase token parallelism in recurrent prepare/tail
projections, especially FFN Down and the other deferred quantized projections.
Attention batching is secondary at P512, though its ordered cost must remain
causal and be rechecked at larger contexts.

## Decision

**KEEP as the production profiling surface.** The profile closes the question
of whether a recurrent scan is the first high-leverage change and provides the
baseline for a larger causal chunk experiment.

## Follow-up

Test a causally valid larger logical chunk with a projection mapping that does
not simply repeat B=4 GEMV microtiles. EXP-0231/0232 show that a grouped MMQ
tile is useful only at a sufficiently large token tile, so the next experiment
must include the recurrent dependency schedule and end-to-end P512 timing.
