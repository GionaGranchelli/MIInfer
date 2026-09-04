# EXP-0096 — M6-B3 Q4_K metadata staging

## Question

Can Q4_K × Q8_K GEMV avoid decoding the same eight scale/min pairs once per
lane and per block by staging block metadata cooperatively in shared memory?

## Baseline

The accepted post-M6-B2 native-generation baseline was:

| Workload | Median | Throughput |
| --- | ---: | ---: |
| TG64 | 14921.8 ms | 4.28904 tok/s |
| TG128 | 30149.9 ms | 4.24545 tok/s |

The Q4 kernel called `q4_k_scale_min` independently for every lane while
processing each block. This affected recurrent and full-attention Q4
projections.

## Candidate

Each 256-thread workgroup now stages up to 32 blocks of Q4 scale/min metadata
and converted `d`/`dmin` values in shared memory. Threads then reuse that
metadata for the unchanged per-element Q4 × Q8 dot product. Larger reductions
are handled in metadata tiles; no weight bytes, quantization format, output
type, or reduction contract changes.

The first implementation attempt used the wrong packed-nibble index and
changed the first generated token. That attempt was rejected immediately,
the index was corrected, and no invalid result was used for the A/B decision.

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
| TG64 median | 4.28904 tok/s | **5.93591 tok/s** | **+38.40%** |
| TG128 median | 4.24545 tok/s | **5.86346 tok/s** | **+38.11%** |

Candidate raw samples:

```text
TG64:  10761.3, 10775.5, 10781.8, 10786.3, 10793.1 ms
TG128: 21712.3, 21792.7, 21830.1, 21853.8, 21860.5 ms
```

The P64 profile changed from approximately 234.2 ms to 169.9 ms total GPU
time, and layer sum changed from approximately 227.1 ms to 162.8 ms. In
layer 0, Q4 `gate_projection` changed from 0.248 ms to 0.127 ms and Q4
`ffn_gate_up` changed from 1.179 ms to 0.622 ms. The Q5
`ssm_output_projection` remained approximately 1.15 ms, isolating the gain to
Q4 metadata reuse.

## Correctness and resources

* Corrected-candidate 16-token smoke: exact first/last tokens and replay PASS.
* TG64 benchmark replay: PASS.
* TG128 benchmark replay: PASS.
* Release CTest: **20/20**.
* Decode-loop allocations: **0**.
* Tracked device bytes after setup and peak: `17,018,706,644`.
* Stable replay fingerprint remained `1853370403272745608`.
* Stable-peak telemetry observed 1725 MHz SCLK and 1000 MHz MCLK.

## Interpretation

This is a repeatable whole-token win from eliminating redundant Q4 metadata
decoding. It does not rely on reducing dispatch count, changing clocks, or
altering the model's quantized arithmetic. The measured profile confirms that
the affected Q4 projection families, not the already-optimized Q5 path, drive
the improvement.

## Decision

**KEEP / M6-B3 production-selected.**

## Follow-up

Refresh the full production profile and compare the resulting 5.9 tok/s path
against the pinned llama.cpp baseline. Select the next candidate from the new
dominant stage; do not repeat rejected workgroup geometries without new
evidence.
