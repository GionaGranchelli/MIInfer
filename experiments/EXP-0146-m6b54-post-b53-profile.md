# EXP-0146 — M6-B54 post-B53 production profile

## Question

What is the current whole-token cost after rejecting the B53 Q4_K×Q8_K FFN
differential, and which production family has the largest remaining measured
opportunity?

## Baseline

The B41 Q4_K×Q8_1 MMVQ path and transposed no-decay-store recurrent update
remain selected. B53's Q4_K×Q8_K alternative is disabled.

## Environment

```text
GPU:       AMD Instinct MI50 / gfx906
Model:     /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Build:     build/mi50-release
Clock:     stable_peak; 1725 MHz SCLK / 1000 MHz MCLK observed
Fixture:   /tmp/m6a273-reference-p12
ROCm:      7.1.52802-9999
Compiler:  Clang 20.0.0.rocm
```

## Commands

```bash
build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  /tmp/m6a273-reference-p12 --profile64
build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  /tmp/m6a273-reference-p12 --bench64
build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  /tmp/m6a273-reference-p12 --bench128
```

## Results

Position-63 profile:

```text
total GPU event: 73.9105 ms/token
layer sum:       70.5859 ms/token
final norm:       0.02304 ms
final Q8:         0.00816 ms
final LM head:    2.52640 ms
final argmax:     0.500639 ms
dispatches:       1,588 (native profile counter not exposed)
allocations:      0
```

Representative stage timings (ms/token):

| Stage | Layer 0 | Layer 1 | Layer 2 | Attention layer 3 |
| --- | ---: | ---: | ---: | ---: |
| QKV / Q projection | 0.183839 | 0.172960 | 0.172000 | 0.207840 |
| FFN Gate/Up | 0.278240 | 0.260640 | 0.271839 | 0.259840 |
| FFN Down | 0.446719 | 0.418240 | 0.445920 | 0.418559 |
| State update / cached attention | 0.119520 | 0.114400 | 0.114080 | 0.139040 cached attention |

The complete 64-layer profile was deterministic and passed its existing
profile gate. FFN Down remains the largest repeated projection family across
the recurrent and full-attention layers; the final LM head is the largest
single non-layer operation.

Native benchmark medians (five reset-separated samples after warmup):

| Workload | Median ms | Median tok/s | Replay | Allocations |
| --- | ---: | ---: | :---: | ---: |
| TG64 | 4604.14 | **13.9005** | PASS | 0 |
| TG128 | 9362.23 | **13.6719** | PASS | 0 |

Raw samples in ms:

```text
TG64:  4599.42, 4604.08, 4604.14, 4607.91, 4622.34
TG128: 9355.16, 9358.60, 9362.23, 9364.72, 9365.67
```

Tracked device bytes after setup and at peak remained
`17,019,965,780`; no decode-loop allocations occurred.

The pinned llama.cpp P1+TG64 control is `21.8576 tok/s` and P1+TG256 is
`21.9405 tok/s` from EXP-0050. This is a token-count-equivalent directional
comparison, not a claim of identical prompt content or harness behavior.

## Interpretation

The current MIInfer path is stable but remains approximately 1.57× slower than
the pinned P1+TG64 llama.cpp control. GPU work continues to explain the
runtime; no allocation, copy, synchronization, or dispatch-count growth is
present. FFN Down is still the largest repeated family, but B52/B53 show that
the tested standalone geometry and Q8_K representation changes are not useful.

The next optimization must therefore target a materially different, measured
whole-pipeline mechanism rather than repeat those rejected projection variants.

## Decision

**MEASUREMENT-ONLY / BASELINE.** Retain B41 production behavior and use this
profile as the post-B53 baseline.

## Follow-up

Select one higher-level FFN execution experiment only after inspecting the
complete Down-input-to-residual dataflow and estimating its removable work.
If no credible work-elimination opportunity exists, profile the next-largest
family instead of repeating standalone Down GEMV tuning.
