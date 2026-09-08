# EXP-0246 — M11-B Q4_K MMQ64 split-4 rejection

## Hypothesis

A single 256-thread workgroup owning four output rows and 64 tokens can
amortize Q4_K weight staging across a full prefill chunk. Four 16-token input
subtiles keep the per-thread accumulator count bounded while preserving the
native Q4_K layout.

## Baseline

Sixteen existing native Q4_K B=4 launches for the exact same 64-token tile.

## Candidate

One launch with four output rows, four cooperating threads per output cell,
Q4_K weights staged once per K tile, and four independent token-group
accumulators.

## Environment

- GPU: AMD Instinct MI50, gfx906
- Model: `Qwen3.8-27B-Q4_K_M.gguf`
- Tensor: `blk.8.ffn_down.weight`, exact Q4_K shape `[17408, 5120]`
- Build: `mi50-release`
- Warmup: 20 baseline/candidate pairs
- Timing: 51 interleaved HIP-event samples per path

## Benchmark

```text
./build/mi50-release/miinfer-q4k-mmq64-split4-bench \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
```

## Correctness

The candidate produced finite output and matched the B=4 baseline with
maximum absolute error `3.57628e-6`.

## Results

| Path | Median | Relative |
| --- | ---: | ---: |
| Sixteen native B=4 launches | 4786.08 us | 1.000x |
| MMQ64 split-4 candidate | 7246.23 us | 0.660x |

## Interpretation

Weight-tile reuse did not compensate for the repeated LDS input staging,
four-way reductions, and lower parallel efficiency of the grouped mapping on
MI50. The candidate is also far below the end-to-end 2.16x improvement
required by the M11-B 100 tok/s gate.

## Decision

**REJECT.** The candidate, temporary benchmark, declaration, and build target
were removed. Production execution is unchanged.

## Follow-up

Do not pursue another native Q4_K remapping of this shape. Any further M11-B
attempt must change the causal chunk dataflow and be measured end-to-end;
standalone grouped projection work is not sufficient.
