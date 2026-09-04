# EXP-0101 — M6-B8 cached-attention Wave64-local reduction

## Question

Can the existing 256-thread cached-attention kernel replace its repeated
256-thread score reduction with four Wave64-local reductions and a four-value
combination, without changing the cache or softmax algorithm?

## Baseline

The accepted native Qwen3.8-27B Q4_K_M path from EXP-0100 used the existing
shared-memory tree reduction. Its stable-peak medians were:

| Workload | Median | Throughput |
| --- | ---: | ---: |
| TG64 | 8,338.45 ms | 7.67529 tok/s |
| TG128 | 16,905.7 ms | 7.57143 tok/s |

The TG128 raw median was 16,952.7 ms. The baseline path had deterministic
replay, zero decode-loop allocations, and `17,018,706,644` device bytes.

## Candidate

The score dot product was changed to reduce within each Wave64 using
`__shfl_down(..., 64)`. Wave leaders stored four partial sums in shared
memory, and lane zero combined them before applying the scale. The existing
cache layout, softmax stages, value reduction, and output contract were left
unchanged.

This was a diagnostic candidate only; it changed floating-point reduction
ordering and was never production-selected.

## Environment

```text
GPU:          AMD Instinct MI50 / gfx906
Clock policy: stable_peak (1725 MHz SCLK / 1000 MHz MCLK observed)
Model:        /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Build:        build/mi50-release
ROCm:         7.1.52802-9999
Compiler:     Clang 20.0.0.rocm
```

## Commands

```bash
build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  /tmp/m6a273-reference --generate16

build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  /tmp/m6a273-reference --bench64

build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  /tmp/m6a273-reference --bench128
```

## Results

| Workload | Baseline throughput | Candidate throughput | Delta |
| --- | ---: | ---: | ---: |
| TG64 | 7.67529 tok/s | 7.67653 tok/s | +0.02% |
| TG128 | 7.57143 tok/s | 7.55040 tok/s | -0.28% |

Candidate raw samples were:

```text
TG64: 8321.64, 8326.50, 8337.10, 8341.97, 8343.75 ms
TG128: 16847.3, 16888.5, 16952.7, 16975.1, 17000.6 ms
```

The 16-token generation check passed with exact replay:

```text
first_token=11
last_token=585
replay=PASS
allocations_during_decode=0
device_bytes_after_setup=17018706644
```

## Correctness

The candidate built successfully and passed the native generation replay and
the Release CTest suite. The suite finished `20/20` after restoring the
accepted implementation, including cached-attention determinism.

## Interpretation

Wave-local score reduction did not produce a repeatable end-to-end benefit.
It was neutral at TG64 and regressed TG128. The small TG64 difference is below
the useful threshold and the candidate does not improve the context-sensitive
workload.

This does not establish that attention is fully optimized; it only rejects
this reduction-order change. The accepted cooperative attention path remains
the production implementation.

## Decision

**REJECT — diagnostic candidate.**

No production code remains from the candidate.

## Follow-up

Do not repeat Wave64 reduction variants without new profiling evidence. Future
attention work should target a materially different mechanism or a measured
whole-token cost, not another local reduction-order change.
