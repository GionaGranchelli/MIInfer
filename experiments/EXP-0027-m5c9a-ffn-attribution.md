# EXP-0027 — M5-C9a production FFN attribution

**Status:** CLOSED — measurement-only; fusion candidate retained
**Milestone:** M5
**Date:** 2026-09-01
**Baseline commit:** `49382d51f518be3dc20d8aaff8e746331177a0e9`
**Candidate:** profiling-only FFN stage counters in the production position audit

## 1. Question

After M5-C8 closed standalone Down GEMV geometry work, where does the current
production decode token spend GPU time inside the complete FFN pipeline, and is
the SwiGLU-to-Down-input quantization boundary large enough to justify a
fusion experiment?

## 2. Method

The production trace-free path was profiled at positions 1 and 64 using the
pinned Qwen3-8B Q4_0 artifact. The audit used the GPU-argmax path, persistent
decode workspace, resident normalization weights, direct layer-output handoff,
and coalesced KV store. The detailed operation events were deferred until the
end of each audit pass; their summed GPU times are attribution data, not clean
throughput measurements.

Command:

```bash
build/mi50-release/miinfer-qwen3-position-audit \
  /path/to/Qwen3-8B-q4_0-b968826d.gguf \
  --positions 1,64 \
  --gpu-argmax \
  --json-output /tmp/mi50-c9a-position-audit.json
```

Environment:

* AMD Instinct MI50 / gfx906
* Release build, HIP 7.1.52802-9999, Clang 20.0.0
* Qwen3-8B Q4_0, model SHA256
  `458634762bea7dbe19f3ce0614465bafd15ee90e815229c427043afcf195d628`
* observed auto-mode clocks were approximately 930 MHz SCLK / 350 MHz MCLK;
  absolute rates are not canonical-clock claims

The new counters add no kernel selection, allocation, copy, synchronization,
or precision behavior. FFN stage timings are collected from the same
production operations already represented by the aggregate profile.

## 3. Full-token attribution

The position audit reported these deferred GPU-event totals:

| Category | P1 ms | P1 % | P64 ms | P64 % | P64 dispatches |
|---|---:|---:|---:|---:|---:|
| Embedding | 0.007 | 0.0% | 0.017 | 0.1% | 1 |
| Normalization | 2.883 | 12.3% | 2.889 | 10.3% | 289 |
| Quantization | 3.782 | 16.1% | 3.759 | 13.4% | 505 |
| Q/K/V projection | 1.666 | 7.1% | 1.658 | 5.9% | 108 |
| O projection | 0.831 | 3.5% | 0.819 | 2.9% | 36 |
| FFN projection | 6.879 | 29.4% | 6.983 | 24.9% | 108 |
| RoPE | 0.551 | 2.4% | 0.550 | 2.0% | 72 |
| Attention | 0.456 | 1.9% | 4.926 | 17.6% | 36 |
| Activation | 0.250 | 1.1% | 0.249 | 0.9% | 36 |
| Residual | 0.497 | 2.1% | 0.496 | 1.8% | 72 |
| Conversion | 2.213 | 9.4% | 2.199 | 7.9% | 324 |
| LM head | 2.879 | 12.3% | 2.910 | 10.4% | 1 |
| Argmax | 0.276 | 1.2% | 0.275 | 1.0% | 1 |
| KV store | 0.260 | 1.1% | 0.263 | 0.9% | 36 |
| **Total deferred GPU events** | **23.430** | **100%** | **27.991** | **100%** | **1,625** |

Clean production wall time was 15.366 ms at P1 and 19.858 ms at P64. The
deferred profile deliberately adds event-recording overhead, so the category
sum must not be substituted for those wall measurements.

## 4. FFN stage attribution

| FFN stage | P1 ms | P1 FFN % | P64 ms | P64 FFN % | Dispatches/token |
|---|---:|---:|---:|---:|---:|
| FFN normalization | 0.883 | 8.9% | 0.885 | 8.9% | 72 |
| Gate input Q8 quantization | 0.534 | 5.4% | 0.533 | 5.3% | 72 |
| Gate projection | 1.900 | 19.2% | 1.881 | 18.9% | 36 |
| Up input Q8 quantization | 0.543 | 5.5% | 0.531 | 5.3% | 72 |
| Up projection | 1.891 | 19.1% | 1.878 | 18.8% | 36 |
| SwiGLU | 0.250 | 2.5% | 0.249 | 2.5% | 36 |
| Down-input Q8 quantization | 0.541 | 5.5% | 0.546 | 5.5% | 72 |
| Down projection | 3.088 | 31.3% | 3.224 | 32.3% | 36 |
| FFN residual | 0.250 | 2.5% | 0.250 | 2.5% | 36 |
| **FFN total** | **9.880** | **100%** | **9.979** | **100%** | **468** |

The production quantization path uses the exact 36-byte Q8 metadata layout;
the stage name is shortened to Q8 for readability. Each FFN layer has
`4096` hidden elements and `12288` intermediate elements. The derived logical
payloads are:

* hidden F32 vector: `4096 × 4 = 16,384` bytes;
* one Q8 block stream for hidden input: `128 × 36 = 4,608` bytes;
* one intermediate F32 vector: `12288 × 4 = 49,152` bytes;
* one intermediate Q8 block stream for Down: `384 × 36 = 13,824` bytes.

These are tensor payload sizes, not hardware-counter measurements. The
profile's measured copy accounting remains the separate aggregate
`589,828` bytes/token, consisting of the 36 layer-input D2D copies plus the
4-byte greedy token result.

## 5. Interpretation

The FFN is still the largest named full-token family. Within it, Down is the
largest single stage at about 3.2 ms/P64, but C8 has already shown that its
standalone GEMV geometry has no supported improvement. The more actionable
boundary is:

```text
SwiGLU (0.249 ms)
    -> intermediate F32 materialization
    -> Down-input Q8 quantization (0.546 ms)
    -> Down GEMV
```

The two stages together account for approximately 0.795 ms at P64 and 108
dispatches per token. A fused SwiGLU-to-Q8 kernel could remove the global
SwiGLU intermediate materialization and reduce this stage chain to one launch
per layer, while preserving the existing Q8 quantization contract. The
maximum directly measured GPU-time opportunity is modest (about 2.8% of the
P64 deferred GPU-event total before accounting for fusion overhead), but the
candidate is sufficiently concrete for one isolated C9b experiment.

There is also a separate reuse opportunity: Gate and Up quantize the same
FFN-normalized input independently, costing about 1.064 ms and 144 dispatches
at P64. That is a promising later candidate, but it is not combined with the
SwiGLU experiment.

## 6. Correctness

The profiling-only build completed cleanly. Release CTest passed 19/19:
8/8 host-only and 11/11 GPU-required tests. The audited path remained
deterministic and produced the existing selected-token sequence:

```text
8, 341, 286, 470, 330, 9707, 11, 330, ...
```

No production kernel, precision policy, memory lifetime, copy path, or
dispatch selection changed.

## 7. Decision

```text
CLOSED — attribution complete; no production optimization in C9a
```

Retain **fused SwiGLU → Down-input Q8 quantization** as the next isolated
candidate (C9b). Do not combine it with Gate/Up quantization reuse, Down GEMV
geometry, or graph capture.

## 8. Follow-up

Implement one fused C9b candidate only if it preserves the exact existing
Q8 metadata/lane contract. Compare it against this production profile and
the clean C6d/C8 baseline using interleaved A/B measurement. If the end-to-end
gain is not repeatable, retain the profile result and move to the separately
measured Gate/Up activation-reuse opportunity.
