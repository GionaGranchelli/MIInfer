# EXP-0104 — M6-B11 Q4_K packed dot4 projections

## Question

Can the scalar Q4_K × Q8_K projection path be replaced by a gfx906 Wave64
packed-dot4 implementation that computes four quantized elements per lane,
while preserving the Q4_K block layout and external inference contract?

## Baseline

The accepted production path from EXP-0100 used one 256-thread workgroup per
output row and decoded one Q4_K element per lane per block. Stable-peak
medians were:

| Workload | Median | Throughput |
| --- | ---: | ---: |
| TG64 | 8,338.45 ms | 7.67529 tok/s |
| TG128 | 16,905.7 ms | 7.57143 tok/s |

## Candidate

The candidate used one Wave64 per output row. Each lane loaded four adjacent
Q4 nibbles and four Q8_K activation values, used the gfx906 signed dot4
instruction for the quantized contribution and the zero-point sum, and then
performed the existing Q4_K scale/minimum epilogue and Wave64 reduction.

The original kernel remained available through `MIINFER_Q4K_DOT4=0` during
the experiment. The candidate became the default after validation.

## Environment

```text
GPU:          AMD Instinct MI50 / gfx906
Clock policy: stable_peak (1725 MHz SCLK / 1000 MHz MCLK observed)
Model:        /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Build:        build/mi50-release
ROCm:         7.1.52802-9999
Compiler:     Clang 20.0.0.rocm
Fixture:      /tmp/m6a273-reference
```

## Results

Final default-path repeated medians:

| Workload | Baseline throughput | Candidate throughput | Delta |
| --- | ---: | ---: | ---: |
| TG64 | 7.67529 tok/s | 8.41092 tok/s | **+9.58%** |
| TG128 | 7.57143 tok/s | 8.32747 tok/s | **+9.99%** |

Candidate raw samples:

```text
TG64: 7595.61, 7604.81, 7609.16, 7613.94, 7622.24 ms
TG128: 15361.8, 15364.0, 15370.8, 15372.2, 15385.9 ms
```

The candidate profile measured `122.581 ms` GPU time at P64 versus
`131.140 ms` for the restored accepted path. The layer-0 FFN Gate/Up stage
changed from approximately `0.621 ms` to `0.507 ms`; profile timings are
supporting evidence, while clean repeated throughput is the selection metric.

## Correctness and resources

* 16-token native generation: PASS with exact replay.
* TG64 benchmark replay: PASS.
* TG128 benchmark replay: PASS.
* Release CTest: **20/20**.
* Decode-loop allocations: **0**.
* Device bytes after setup and peak: `17,018,706,644`.
* Existing external observable checks at available positions remain within the
  established contract; the fixture lacked the requested later checkpoint.

The packed path changes floating-point reduction ordering, so the old
MIInfer internal fingerprint is not used as the correctness authority. Native
replay and the established external observable envelope remain required.

## Interpretation

Packing four Q4 nibbles and four Q8 values into gfx906 dot4 operations removes
substantial scalar decode/accumulation work across the recurrent Q4 projections.
The gain is repeatable at both tested generation lengths and is large enough
to justify production selection.

## Decision

**KEEP / production-selected.**

## Follow-up

Refresh the post-promotion profile before selecting the next optimization. The
accepted path is still well below the pinned llama.cpp baseline, so subsequent
work should target the next largest measured whole-token family rather than
returning to scalar Q4 decode tweaks.
