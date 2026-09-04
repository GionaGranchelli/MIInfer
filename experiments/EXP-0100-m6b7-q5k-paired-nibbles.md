# EXP-0100 — M6-B7 Q5_K paired-nibble decoding

## Question

Can the Q5_K × Q8_K recurrent output projection decode the two subgroups
encoded in each packed `ql` byte together, reducing redundant `ql`/`qh`
loads while preserving the existing integer dot contract?

## Baseline

The accepted subgroup-structured Q5 path from `8f911aa` measured:

| Workload | Median | Throughput |
| --- | ---: | ---: |
| TG64 | 8558.67 ms | 7.47780 tok/s |
| TG128 | 17370.2 ms | 7.36892 tok/s |

## Candidate

The Q5 inner loop now processes four pairs of 32-element subgroups. One
packed `ql` byte supplies both nibbles and one `qh` load supplies both high
bits; the existing scale/minimum arrays, integer partial sums, and float
epilogue remain unchanged.

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
| TG64 median | 7.47780 tok/s | **7.67529 tok/s** | **+2.64%** |
| TG128 median | 7.36892 tok/s | **7.57143 tok/s** | **+2.75%** |

Candidate raw samples:

```text
TG64:  8323.33, 8333.09, 8338.45, 8338.80, 8340.31 ms
TG128: 16784.6, 16851.7, 16905.7, 16932.5, 16944.7 ms
```

The P64 profile changed from approximately 135.4 ms to 133.8 ms total GPU
time. The profiled layer-0 Q5 `ssm_output_projection` changed from 0.405 ms to
0.324 ms.

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

Pairing the Q5 low/high nibbles removes redundant packed-data loads and gives
a repeatable whole-token improvement. The profile change is localized to the
Q5 recurrent output projection, with no model, state, or dispatch redesign.

## Decision

**KEEP / M6-B7 production-selected.**

## Follow-up

Refresh the full production profile against the pinned llama.cpp baseline.
The next optimization must be selected from the remaining dominant stages.
