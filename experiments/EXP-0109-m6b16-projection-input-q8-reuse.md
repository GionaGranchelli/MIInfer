# EXP-0109 — M6-B16 projection-input Q8_K reuse

## Question

Can repeated quantization of the same normalized activation be removed by
sharing the existing Q8_K buffer between serialized projection consumers?

## Hypothesis

The recurrent QKV/gate and FFN Gate/Up projections, and the full-attention Q/K/V
and FFN Gate/Up projections, consume the same normalized source and can reuse
one byte-identical Q8_K representation. This should remove duplicate quantizer
dispatches and writes without changing any projection arithmetic.

## Baseline and candidate

Baseline: separate quantization, selected with
`MIINFER_REUSE_PROJECTION_Q8=0`.

Candidate: quantize the first consumer's input and reuse the existing Q8_K
buffer for subsequent serialized consumers, selected by default. The candidate
changes no quantizer, projection kernel, representation, precision boundary,
or scheduling concurrency.

## Environment

```text
GPU: AMD Instinct MI50 / gfx906
Clock policy: stable_peak (1725 MHz SCLK / 1000 MHz MCLK observed)
Model: /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Build: build/mi50-release
ROCm: 7.1.52802-9999
Compiler: Clang 20.0.0.rocm
Fixture: /tmp/m6a273-reference-p12
```

## Commands

```bash
MIINFER_REUSE_PROJECTION_Q8=1 build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  /tmp/m6a273-reference-p12 --generate16
MIINFER_REUSE_PROJECTION_Q8=1 build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  /tmp/m6a273-reference-p12 --prefix64-observable-contract
MIINFER_REUSE_PROJECTION_Q8=0 ... --bench64
MIINFER_REUSE_PROJECTION_Q8=1 ... --bench64
MIINFER_REUSE_PROJECTION_Q8=0 ... --bench128
MIINFER_REUSE_PROJECTION_Q8=1 ... --bench128
```

## Correctness

* Native 16-token generation passed with replay `PASS` and zero decode
  allocations; first token `11`, last token `46194`.
* The 64-layer observable run completed through P64 with matching reported
  reference decisions and the accepted margin-aware external contract.
* Release CTest: `20/20 PASS`.
* No new state/KV, NaN/Inf, or host round-trip issue was observed.

## Results

Five-sample serial medians at stable_peak:

| Workload | Separate median | Shared median | Change |
| --- | ---: | ---: | ---: |
| TG64 ms | 7181.18 | 7125.95 | -0.77% |
| TG64 tok/s | 8.91219 | 8.98126 | +0.78% |
| TG128 ms | 14504.50 | 14402.60 | -0.70% |
| TG128 tok/s | 8.82486 | 8.88730 | +0.71% |

Both paths reported replay `PASS`, zero decode allocations, and
`17018712404` tracked/peak device bytes. The native profile harness reports
dispatches as unknown; the source-level candidate removes one Q8_K quantizer
launch for each repeated consumer group while retaining the same serialized
projection sequence.

## Interpretation

The candidate removes real duplicate work and is positive on both required
generation workloads. The gain is below the 2% strong-candidate guideline but
is repeatable, exact under the existing external observable contract, and has
low implementation complexity and no additional VRAM.

## Decision

**KEEP; production-selected.** Shared projection-input reuse is the default.
Set `MIINFER_REUSE_PROJECTION_Q8=0` to run the former separate path for A/B
comparison.

## Follow-up

Refresh the full production profile before selecting the next optimization.
