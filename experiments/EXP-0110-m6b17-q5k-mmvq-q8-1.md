# EXP-0110 — M6-B17 Q5_K×Q8_1 MMVQ recurrent projection

## Question

Can the recurrent `ssm_out` Q5_K×Q8_K projection use a gfx906 MMVQ-style
Q5_K×Q8_1 decomposition to reduce its measured GPU cost without changing the
accepted external observable contract?

## Hypothesis

The recurrent output projection is a repeated, high-cost Q5_K path. The pinned
gfx906 reference uses a 128-thread MMVQ decomposition for Q5_K×Q8_1. Applying
that mechanism to the exact `kInner=6144`, `kHidden=5120` projection should
improve the recurrent-layer bottleneck.

## Baseline and candidate

The baseline is the production Q5_K×Q8_K path, selected with
`MIINFER_Q5K_Q8_1_MMVQ=0`. The candidate is selected by default and performs
the existing FP32 input quantization to canonical Q8_1 blocks, followed by a
128-thread/output-row Q5_K×Q8_1 MMVQ kernel. Gate/Up/Down projections and all
other recurrent operations are unchanged.

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
MIINFER_Q5K_Q8_1_MMVQ=0 build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  /tmp/m6a273-reference-p12 --bench64
MIINFER_Q5K_Q8_1_MMVQ=1 build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  /tmp/m6a273-reference-p12 --bench64
MIINFER_Q5K_Q8_1_MMVQ=0 ... --bench128
MIINFER_Q5K_Q8_1_MMVQ=1 ... --bench128
MIINFER_Q5K_Q8_1_MMVQ=1 ... --generate16
MIINFER_Q5K_Q8_1_MMVQ=1 ... --prefix64-observable-contract
MIINFER_Q5K_Q8_1_MMVQ=1 ... --profile64
```

## Correctness

* Candidate native 16-token generation passed with replay `PASS`, first token
  `11`, last token `585`, and zero decode allocations.
* The 64-layer observable run completed through P64 with high logit cosine,
  matching top-5 sets, and the accepted margin-aware external contract. The
  known P2 low-margin choice remains diagnostic rather than an internal-byte
  requirement.
* Release CTest: `20/20 PASS`.
* Device bytes after setup and peak: `17019044180`; no decode allocation was
  observed. The extra Q8_1 scratch is about 331 KiB across the recurrent layers.

## Results

Serial process medians, each process containing five timed samples:

| Workload | Control median | Candidate median | Change |
| --- | ---: | ---: | ---: |
| TG64 ms | 7130.97 | 6431.70 | -9.81% |
| TG64 tok/s | 8.97493 | 9.95072 | +10.87% |
| TG128 ms | 14392.70 | 13057.30 | -9.28% |
| TG128 tok/s | 8.89340 | 9.80298 | +10.23% |

The candidate profile changed total GPU time from `113.235 ms` to
`101.994 ms`, and layer-sum time from `109.992 ms` to `98.7055 ms`. In the
representative recurrent layer, `ssm_output_projection` fell from `0.3064 ms`
to `0.075519 ms` (-75.4%). Other stage times remained in the same range.

## Interpretation

The Q5_K×Q8_1 MMVQ decomposition targets the measured recurrent output
projection and produces a large, repeatable whole-token improvement. The
candidate adds one small persistent Q8_1 scratch buffer per recurrent layer,
does not allocate in the decode loop, and does not alter unrelated kernels.

This is an architectural-contract experiment: its internal Q8 representation
differs from the former Q8_K path, so old MIInfer intermediate bytes and native
token identity are diagnostic only. External observable checkpoints remain
within the accepted M6 envelope.

## Decision

**KEEP; production-selected.** The Q5_K×Q8_1 MMVQ route is now the default.
Set `MIINFER_Q5K_Q8_1_MMVQ=0` to run the former Q5_K×Q8_K control.

## Follow-up

Refresh the full post-B17 profile before selecting the next optimization. Do
not sweep additional Q5 geometries without new evidence.
