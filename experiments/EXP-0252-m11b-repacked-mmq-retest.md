# EXP-0252 — Repacked MMQ64 retest on the production Q4_K down shape

## Hypothesis

A three-plane Q4_K repack with a 64-row x 64-token skinny-GEMM tile can expose
token parallelism more effectively than the production Wave64 row GEMV during
grouped prefill.

## Motivation

The M11-B profile leaves the deferred recurrent/FFN tail as the dominant
prefill cost. A prior external-style prototype reported a possible gain on a
related shape, so this retest isolates the same mechanism against the current
production native B4 kernel and exact Qwen3.8-27B Q4_K_M FFN-down dimensions.

## Baseline

`launch_q4k_wave_gemv_batched4` using the existing `Q4KWaveTile` layout,
launched once per four-token group.

## Candidate

An isolated 256-thread kernel using:

- three-plane Q4_K weights: nibble, scale/minimum, and d/dmin;
- 64 output rows x 64 input tokens per block;
- BK=4, TM=4, TN=4 register microtiles;
- LDS staging for the current four K subblocks;
- exact int8 activation sums for the Q4_K minimum term.

The candidate was not connected to the runtime.

## Environment

- GPU: AMD Instinct MI50, gfx906
- Model: `/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf`
- Tensor: exact Q4_K FFN-down shape `[17408,5120]`
- Activation: Q8_1
- Build: `mi50-release`
- Baseline source commit: `4c1bc20`

## Benchmark

```text
rtk ./build/mi50-release/miinfer-m11b-repacked-mmq-bench \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
```

51 interleaved A/B rounds were measured for each batch after warm-up and
device synchronization. Times are median microseconds for the whole batch.

## Correctness

Candidate output matched the native single-token reference within `6.5e-6`
maximum absolute error across the tested batches.

An initial version used the stored Q8_1 half partial sum for the minimum term
and differed by up to `1.6e-2`; exact int8 summation removed that error.

## Results

| Batch | Native B4 (us) | Repacked MMQ64 (us) | Candidate / baseline |
|---:|---:|---:|---:|
| 4 | 322.72 | 4809.60 | 14.88x |
| 16 | 1217.60 | 4803.84 | 3.95x |
| 64 | 4779.20 | 4885.44 | 1.022x |

The B64 candidate is `0.978x` the baseline speed; B4 and B16 are substantially
slower because the tile computes a full 64-token wave even when the batch is
smaller.

## Profiling

No production integration or end-to-end profile was run. The standalone result
does not provide a credible path to the M11-B P512 gate.

## Interpretation

Concurrent row/token execution does not overcome the extra repack traffic,
register work, and exact minimum-term reduction on the current MI50 shape.
The candidate is near parity only at B64 and is not useful for the production
P512 path, whose logical chunks are B4 at the deferred tail.

## Decision

REJECT

## Follow-up

Do not retain the layout or kernel. Any future causal grouped-dataflow test
must reduce the B4/B16 tail cost directly rather than widening this MMQ64 tile.
