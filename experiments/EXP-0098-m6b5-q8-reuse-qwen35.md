# EXP-0098 — M6-B5 Qwen3.8 Q8_K activation reuse

## Question

Can identical normalized activations be quantized once and reused by
recurrent `qkv`/`gate`, FFN `gate`/`up`, and full-attention Q/K/V projections?

## Baseline and candidate

Baseline was the accepted Q4 metadata-staged/Q5 scale-hoisted path from
`d1d9fef`. The candidate skipped duplicate Q8_K quantization calls when the
consumer projections received the same source activation, without changing
the quantizer or GEMV kernels.

## Environment

```text
GPU:          AMD Instinct MI50 / gfx906
Model:        /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Clock policy: stable_peak (1725 MHz SCLK / 1000 MHz MCLK observed)
Build:        build/mi50-release
Fixture:      /tmp/m6a273-reference
```

## Results

| Workload | Baseline | Candidate | Change |
| --- | ---: | ---: | ---: |
| TG64 median | 5.93591 tok/s | 5.95677 tok/s | +0.35% |
| TG128 median | 5.86346 tok/s | 5.88289 tok/s | +0.33% |

Both benchmark replays passed, allocations remained zero, and device bytes
remained `17,018,706,644`. The gains are below the pre-registered useful
end-to-end threshold and within the observed run spread.

## Decision

**REJECT for production selection.** The reuse is semantically safe but does
not produce a repeatable useful whole-token improvement on this workload.

## Follow-up

Restore separate quantization and profile the accepted path again before
choosing the next larger cost center.
