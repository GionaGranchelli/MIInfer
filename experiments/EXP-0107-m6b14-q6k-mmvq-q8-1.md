# EXP-0107 — M6-B14 Q6_K MMVQ-style Q8_1 LM-head candidate

## Question

Can the Q6_K × Q8_1 LM-head path use llama.cpp's gfx906 MMVQ-style
decomposition and materially reduce the final vocabulary projection cost?

## Baseline

The accepted production path is Q6_K × Q8_K. The prior scalar Q6_K × Q8_1
compatibility path was rejected in EXP-0106 because representation matching
alone made P64 11.28% slower.

## Candidate

The opt-in candidate uses canonical Q8_1 blocks and a llama.cpp-derived GCN
MMVQ mapping: two Wave64s (128 threads) per output row, four Q6_K blocks
visited per iteration, Q6 sub-block unpacking, and packed int8 dot products.
The Q8_1 quantizer and existing Q6_K/Q8_1 representation are otherwise
unchanged. The control remains the default; enable the candidate with
`MIINFER_LM_Q8_1_MMVQ=1`.

The first prototype omitted the high-bit-plane shift required by the pinned
MMVQ unpack formula. That was corrected before the results below; the initial
corrupt output is retained here as implementation evidence, not as a
benchmark result.

## Environment

```text
GPU: AMD Instinct MI50 / gfx906
Clock policy: stable_peak (1725 MHz SCLK / 1000 MHz MCLK observed)
Model: /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Build: build/mi50-release
ROCm: 7.1.52802-9999
Compiler: Clang 20.0.0.rocm
Fixture: /tmp/m6a273-reference-p12 (complete temporary observable fixture)
```

## Correctness

```text
Release CTest: 20/20 PASS
generate16: PASS; replay PASS; allocations during decode: 0
generate64: PASS; replay PASS; allocations during decode: 0
```

The complete temporary observable fixture was rerun after the high-bit-plane
correction. The candidate produced 62/64 teacher-forced argmax matches; the
only differences were the already-adjudicated low-margin P2 and P12 choices.
The late checkpoints, including P64, completed successfully. At P64 the
candidate had cosine `0.999616`, top-5 overlap `5/5`, matching argmax `8719`,
and deterministic replay remained exact. The two non-matching positions retain
the established margin-aware external contract: the winner is not robust under
the measured logit perturbation and top-k membership remains preserved.

At the earlier external observable checkpoints, the candidate produced:

* P1 cosine `0.999851`, top-5 overlap `4/5`, reference argmax match;
* P2 cosine `0.999831`, top-5 overlap `5/5`, `1318 → 1044`, with the
  reference winner at GPU rank 2 and a non-robust reference margin;
* P4 cosine `0.999841`, top-5 overlap `5/5`, reference argmax match;
* P8 cosine `0.999870`, top-5 overlap `5/5`, reference argmax match.

The complete fixture run also preserved the existing internal state diagnostics
and deterministic replay; those strict state-envelope warnings are not the
external observable acceptance authority established by A27.

## Benchmark

Same-build serial P64 benchmark runs on the same MI50:

| Path | Median ms | Median tok/s |
| --- | ---: | ---: |
| Q6_K × Q8_K control | 7612.10 | 8.40767 |
| Q6_K × Q8_1 MMVQ candidate | 7371.34 | 8.68228 |

The candidate improves end-to-end throughput by `3.26%` and reduces median
decode time by `240.76 ms` over 64 generated tokens. A separate post-fix
64-token generation measured `8.69003 tok/s` on its replay run.

The P64 GPU profile reports:

```text
control final LM-head event: approximately 6.19 ms
MMVQ candidate final LM-head event: 2.44912 ms
MMVQ candidate total GPU event: 119.064 ms
allocations during profile: 0
```

## Interpretation

Unlike EXP-0106, this candidate tests the representation together with the
specific access/reduction strategy used by the pinned gfx906 MMVQ path. The
candidate is a repeatable positive end-to-end result and clears the complete
temporary external observable run under the accepted margin-aware contract.

## Decision

**KEEP; production-selected.** The Q6_K×Q8_1 MMVQ-style path is now the
Qwen3.8 default. Set `MIINFER_LM_Q8_1_MMVQ=0` to run the former Q6_K×Q8_K
control for A/B comparison.

## Follow-up

Refresh the post-selection benchmark/profile and retain the control toggle for
future regressions.
