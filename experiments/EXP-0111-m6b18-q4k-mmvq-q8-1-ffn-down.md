# EXP-0111 — M6-B18 Q4_K×Q8_1 MMVQ FFN Down

## Question

Can the measured Q4_K FFN Down projection use a gfx906 MMVQ-style Q4_K×Q8_1
path to reduce whole-token decode time?

## Hypothesis

Post-B17 profiling leaves FFN Down as a large repeated projection. The pinned
gfx906 reference provides a 128-thread Q4_K×Q8_1 MMVQ decomposition. Applying
that mechanism only to Q4 FFN Down should improve the exact
`kHidden=5120`, `kFfnInner=17408` shape without changing other projections.

## Baseline and candidate

The baseline is the production Q4_K×Q8_K packed-dot4 path, selected with
`MIINFER_Q4K_Q8_1_MMVQ=0`. The candidate is selected by default and performs
the existing FP32 activation quantization to canonical Q8_1 blocks, followed
by a 128-thread/output-row Q4_K×Q8_1 MMVQ kernel. It is applied only when the
FFN Down weight is Q4_K. Gate, Up, recurrent output, attention, and LM-head
paths are unchanged.

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
MIINFER_Q4K_Q8_1_MMVQ=0 build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  /tmp/m6a273-reference-p12 --bench64
MIINFER_Q4K_Q8_1_MMVQ=1 build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  /tmp/m6a273-reference-p12 --bench64
MIINFER_Q4K_Q8_1_MMVQ=0 ... --bench128
MIINFER_Q4K_Q8_1_MMVQ=1 ... --bench128
MIINFER_Q4K_Q8_1_MMVQ=1 ... --generate16
MIINFER_Q4K_Q8_1_MMVQ=1 ... --prefix64-observable-contract
MIINFER_Q4K_Q8_1_MMVQ=1 ... --profile64
```

## Correctness

* Candidate native 16-token generation passed with replay `PASS`, first token
  `11`, last token `585`, and zero decode allocations.
* The 64-layer observable run completed through P64 under the accepted external
  observable contract; final logits retained high cosine and matching top-5
  sets. The known low-margin P2 decision remains diagnostic.
* Release CTest: `20/20 PASS`.
* Candidate setup/peak device bytes: `17019965780`; no decode allocation was
  observed. The Q8_1 scratch is persistent and shared per layer.

## Results

Three serial process medians, each process containing five timed samples:

| Workload | Control median | Candidate median | Change |
| --- | ---: | ---: | ---: |
| TG64 ms | 6435.11 | 6289.26 | -2.27% |
| TG64 tok/s | 9.94544 | 10.1761 | +2.32% |
| TG128 ms | 13046.10 | 12769.70 | -2.12% |
| TG128 tok/s | 9.81133 | 10.0237 | +2.16% |

The candidate profile measured `100.021 ms` total GPU time versus `102.133 ms`
for control. The representative `ffn_down` stage measured `0.433279 ms`
versus `0.418239 ms` in that pair; the stage timer is noisy and does not show
the whole-token win by itself. Repeated clean throughput is the selection
metric.

## Interpretation

The candidate gives a repeatable useful whole-token improvement at both TG64
and TG128 while changing only the Q4 FFN Down representation and kernel. It
adds persistent Q8_1 scratch but no decode-loop allocations. The result is
large enough to meet the project's useful-candidate threshold.

## Decision

**KEEP; production-selected.** The Q4_K×Q8_1 MMVQ route is now the default.
Set `MIINFER_Q4K_Q8_1_MMVQ=0` to run the former Q4_K×Q8_K control.

## Follow-up

Refresh the complete post-B18 profile before selecting the next optimization.
Do not sweep additional Q4 geometries without new evidence.
