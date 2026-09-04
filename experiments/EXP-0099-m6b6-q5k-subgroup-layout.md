# EXP-0099 — M6-B6 Q5_K subgroup-structured dot loop

## Question

Can the Q5_K × Q8_K recurrent output projection avoid repeated per-element
index arithmetic by traversing its eight 32-element subgroups explicitly?

## Baseline

The accepted Q4 metadata-staged/Q5 scale-hoisted path from `d1d9fef` measured:

| Workload | Median | Throughput |
| --- | ---: | ---: |
| TG64 | 10781.8 ms | 5.93591 tok/s |
| TG128 | 21830.1 ms | 5.86346 tok/s |

The Q5 kernel performed division/modulo-based group and packed-index
calculation for every one of its 256 elements.

## Candidate

The kernel now iterates `group=0..7` and `index=0..31`, computing each
subgroup's packed-weight base, high-bit shift, and nibble selection once per
group while retaining the existing integer partial sums, scale/minimum
hoisting, Q8_K input, and final float accumulation contract.

## Environment

```text
GPU:          AMD Instinct MI50 / gfx906
Model:        /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Model bytes:  17,106,775,008
Clock policy: stable_peak (1725 MHz SCLK / 1000 MHz MCLK observed)
Build:        build/mi50-release
ROCm:         7.1.52802-9999
Compiler:     Clang 20.0.0.rocm
Fixture:      /tmp/m6a273-reference
```

## Commands

```bash
build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  /tmp/m6a273-reference --bench64
build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  /tmp/m6a273-reference --bench128
build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  /tmp/m6a273-reference --profile64
ctest --test-dir build/mi50-release --output-on-failure
```

Each benchmark warms up, runs five reset-separated timed samples, and checks
all timed token/state fingerprints against the warmup.

## Results

| Workload | Baseline | Candidate | Change |
| --- | ---: | ---: | ---: |
| TG64 median | 5.93591 tok/s | **7.47780 tok/s** | **+25.98%** |
| TG128 median | 5.86346 tok/s | **7.36892 tok/s** | **+25.68%** |

Candidate raw samples:

```text
TG64:  8549.01, 8555.37, 8558.67, 8568.57, 8569.41 ms
TG128: 17274.0, 17317.4, 17370.2, 17387.0, 17402.7 ms
```

The P64 profile changed from approximately 169.9 ms to 135.4 ms total GPU
time, and layer sum changed from approximately 162.8 ms to 128.3 ms. In
layer 0, `ssm_output_projection` changed from 1.154 ms to 0.405 ms, a
64.9% reduction. The Q4 stages remained at their accepted post-EXP-0096
times.

## Correctness and resources

* 16-token smoke: exact first/last tokens and replay PASS.
* TG64 benchmark replay: PASS.
* TG128 benchmark replay: PASS.
* Release CTest: **20/20**.
* Decode-loop allocations: **0**.
* Tracked device bytes after setup and peak: `17,018,706,644`.
* Stable replay fingerprint remained `1853370403272745608`.
* Stable-peak telemetry observed 1725 MHz SCLK and 1000 MHz MCLK.

## Interpretation

The explicit subgroup traversal removes repeated integer index arithmetic from
the dominant Q5 projection while preserving its integer accumulation and
floating-point epilogue. The large, repeatable end-to-end gain and matching
profile attribution identify this as real device-work reduction rather than
dispatch or clock noise.

## Decision

**KEEP / M6-B6 production-selected.**

## Follow-up

Refresh the full profile against the pinned llama.cpp control. The next target
must be selected from the remaining Qwen3.8 recurrent and whole-token costs;
do not infer it from the old pre-EXP-0099 ranking.
