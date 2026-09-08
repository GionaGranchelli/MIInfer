# EXP-0234 — M11-B causal recurrent-core B=4 batch

## Hypothesis

The layer-major path already prepares four recurrent tokens' normalized
query/key/value, beta, decay, and gate inputs before the deferred tail. A
single workgroup can apply those four recurrent updates in token order while
keeping each state row in registers, reducing state traffic and dispatches
without changing causal semantics.

## Candidate

The candidate stages the six per-token recurrent inputs with one GPU staging
kernel, then launches one fused B=4 recurrent-core kernel per layer chunk.
That kernel loads the persistent state row once, applies tokens 0 through 3
sequentially, performs the existing head normalization and SiLU gate for each
token, and writes four gated outputs plus the final state.

The old per-token path remains available with
`MIINFER_PREFILL_RECURRENT_BATCH=0`. The candidate is enabled by default only
when the existing opt-in layer-major prefill is enabled.

## Correctness

The P17 candidate and control produced the same one-token `hello`
continuation. P128 and P513 candidate runs completed one-token continuation
without non-finite output. The candidate uses the same state update arithmetic
and reduction order as the existing fused core, with only the causal token
loop moved inside one workgroup. Full CTest passed `21/21`.

## Environment

- GPU: AMD Instinct MI50, gfx906
- Model: `Qwen3.8-27B-Q4_K_M.gguf`
- Build: `mi50-release`
- ROCm: 6.4.0 / LLVM 20
- Prompt: repeated `hello` tokens
- `MIINFER_PREFILL_LAYER_MAJOR=1`
- A/B control: `MIINFER_PREFILL_RECURRENT_BATCH=0`

## Results

Production-shaped P513 runs, three process repetitions per side:

| Path | Runs (ms) | Median | PP tok/s |
| --- | --- | ---: | ---: |
| Existing layer-major control | 11192.28 / 11184.44 / 11186.58 | 11186.58 | 45.86 |
| Batched recurrent core | 11091.71 / 11178.23 / 11106.30 | 11106.30 | 46.19 |

Candidate speedup: `1.0072×` (`+0.72%`). Supporting P128 pairs were
`2687.14 / 2774.20 ms` candidate and `2820.54 / 2763.23 ms` control. The
one-token decode measurements remained within normal noise (`1.56–1.61 ms`).

The six recurrent staging buffers add approximately `12.1 MiB` across the 48
recurrent layers. They are allocated once with the existing layer-major
workspace and are not allocated or used by decode.

## Interpretation

The gain is small because the recurrent state update is only a minor fraction
of the full prefill path, but the result validates a causally correct larger
unit of recurrent work and removes five of six per-token staging commands. It
does not change the dominant quantized projection floor established by
EXP-0233.

## Decision

**KEEP** inside the opt-in layer-major prefill path. The token-major path and
decode path remain unchanged. The M11-B 100 tok/s gate remains open.

## Follow-up

Retain the B=4 causal core as a supporting optimization while pursuing a new
whole-pipeline dataflow; do not treat its small gain as evidence that the
projection ceiling has moved.
