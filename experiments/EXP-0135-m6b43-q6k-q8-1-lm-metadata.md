# EXP-0135 — M6-B43 Q6_K×Q8_1 LM-head metadata staging

## Question

Can the Q6_K×Q8_1 MMVQ LM head reuse each row's `d` and scale metadata from
LDS while retaining its existing Q6 payload loads, arithmetic, reduction, and
external output contract?

## Candidate

The diagnostic candidate staged the 20 Q6_K blocks' per-row `d` and 16 signed
scales in LDS for the `[248320, 5120]` LM-head matvec. Q6 low/high payload bytes,
Q8_1 inputs, dot4 operations, accumulation order, and output handling were
unchanged. The candidate was opt-in and never production-selected.

## Environment

```text
GPU:       AMD Instinct MI50 / gfx906
Clock:     stable_peak (1725 MHz SCLK / 1000 MHz MCLK observed)
Model:     /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Build:     build/mi50-release
Fixture:   /tmp/m6a273-reference-p12
```

## Correctness

Native 16-token generation passed exact replay with zero decode allocations.
Device usage was unchanged at `17,019,965,780` bytes.

## Results

Serial same-build medians, candidate versus current B41 control:

| Workload | Control tok/s | Candidate tok/s | Change |
| --- | ---: | ---: | ---: |
| TG64 | 13.8960 | 13.9143 | +0.13% |
| TG128 | 13.6772 | 13.6857 | +0.06% |

The differences are below the useful threshold and within run dispersion.
The parallel benchmark attempt exited during setup with out-of-memory before
producing measurements and is not used as evidence.

## Decision

**REJECT.** Metadata-only LDS staging does not produce a repeatable whole-token
gain. B41's decoded Q4_K metadata path and the existing Q6_K×Q8_1 MMVQ LM head
remain unchanged.

## Follow-up

Do not revisit this metadata variant without new profiling evidence. The next
candidate must target a different measured source of whole-token cost.
