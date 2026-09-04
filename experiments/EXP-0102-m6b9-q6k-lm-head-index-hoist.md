# EXP-0102 — M6-B9 Q6_K LM-head index hoisting

## Question

Can the Q6_K × Q8_K LM-head GEMV avoid recomputing Q6 block index and scale
group arithmetic inside its 20-block-per-row loop without changing its
quantized dot-product contract?

## Baseline

The accepted production path from EXP-0100 used `q6_value()` for each block
and lane. Stable-peak medians were:

| Workload | Median | Throughput |
| --- | ---: | ---: |
| TG64 | 8,338.45 ms | 7.67529 tok/s |
| TG128 | 16,905.7 ms | 7.57143 tok/s |

## Candidate

The Q6 lane decomposition, low/high-bit indices, scale-group index, and high
bit shift were computed once before the block loop. The loop then performed
the same Q6 decode, scale application, integer-to-float conversion, and
accumulation as the production helper.

No quantization format, output representation, reduction structure, or
production model path was otherwise changed.

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

| Workload | Baseline throughput | Candidate throughput | Delta |
| --- | ---: | ---: | ---: |
| TG64 | 7.67529 tok/s | 7.68568 tok/s | +0.14% |
| TG128 | 7.57143 tok/s | 7.55957 tok/s | -0.16% |

Candidate raw samples:

```text
TG64: 8312.44, 8320.46, 8327.17, 8332.88, 8336.42 ms
TG128: 16810.5, 16880.0, 16932.2, 16942.9, 16963.3 ms
```

The 16-token native generation check passed with exact replay:

```text
first_token=11
last_token=585
replay=PASS
allocations_during_decode=0
device_bytes_after_setup=17018706644
```

## Correctness

The candidate built successfully and passed the native generation replay.
Production source was restored before the regression suite; the Release CTest
suite then passed `20/20`, including cached-attention determinism.

## Interpretation

Hoisting Q6 index arithmetic does not produce a repeatable whole-token gain.
The TG64 difference is below the project's useful threshold, and the longer
TG128 workload regresses. The candidate is therefore not worth retaining for
the current Qwen3.8 workload.

## Decision

**REJECT — diagnostic candidate.**

No candidate code remains in the production path.

## Follow-up

Do not repeat local Q6 index variants without new instruction or kernel
profiling evidence. The next optimization should target a measured cost at the
whole-token level or a materially different LM-head mechanism.
