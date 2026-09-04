# EXP-0095 — M6-B2 Q5_K scale/min unpack hoisting

## Question

Can the dominant Qwen3.8 recurrent Q5_K × Q8_K projection be accelerated by
removing repeated unpacking of the same eight scale/min pairs, without
changing the quantized arithmetic contract?

## Baseline

The production Q5_K × Q8_K GEMV recomputed `q5_k_scale_min` for every one of
the 256 elements in each Q5_K block, although each block has only eight scale
and minimum pairs. Clean repeated native-generation baselines were:

| Workload | Median | Throughput |
| --- | ---: | ---: |
| TG64 | 19029.1 ms | 3.36327 tok/s |
| TG128 | 38289.1 ms | 3.34299 tok/s |

## Candidate

The Q5_K kernel now unpacks each block's eight scale/min pairs once into
private arrays, then reuses those values through the existing per-element dot
product and minimum correction. Weight bytes, Q8_K activations, block layout,
accumulation, and output conversion are unchanged.

An alternate 64-thread/one-wave geometry was also checked during the same
investigation. It slowed the 16-token generation smoke test by about 11% and
was rejected; it is not part of the selected change.

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
| TG64 median | 3.36327 tok/s | **4.28904 tok/s** | **+27.53%** |
| TG128 median | 3.34299 tok/s | **4.24545 tok/s** | **+27.00%** |

Candidate raw samples:

```text
TG64:  14894.7, 14895.7, 14921.8, 14953.5, 14967.2 ms
TG128: 29995.0, 30119.8, 30149.9, 30162.2, 30174.7 ms
```

The P64 profile changed from approximately 297.9 ms total GPU time to 234.2
ms. In the profiled recurrent layer 0, `ssm_output_projection` fell from
approximately 2.49 ms to 1.16 ms; the surrounding stage structure remained
unchanged.

## Correctness and resources

* 16-token smoke replay: PASS.
* TG64 benchmark replay: PASS.
* TG128 benchmark replay: PASS.
* Release CTest: **20/20**.
* Decode-loop allocations: **0**.
* Tracked device bytes after setup and peak: `17,018,706,644`.
* No quantization format, weight bytes, state layout, or model behavior was
  intentionally changed.

## Interpretation

The result is a large, repeatable win in the measured dominant recurrent
projection. The improvement is not explained by dispatch removal or clock
changes: the candidate changes only redundant per-block scale/min unpacking,
and the device footprint and replay behavior are unchanged.

## Decision

**KEEP / M6-B2 production candidate.**

The hoisted Q5_K scale/min values are selected in the production kernel. The
64-thread geometry variant remains rejected.

## Follow-up

Refresh the full production profile and compare against the pinned llama.cpp
baseline. Do not assume the next target; rank the remaining recurrent and
whole-token costs from the new profile.
