# EXP-0157 — M6-B65 Q4_K×Q8_1 packed-input inner loop

## Question

Does explicitly packing the two Q8 words and scales into local values improve
the existing decoded-metadata Q4_K×Q8_1 two-row kernel?

## Candidate

The candidate preserves the production 256-thread/two-row mapping, Q4_K
decoded metadata, Q8_1 representation, dot4 arithmetic, and reduction. It
loads both Q8 parts into local arrays before the dot calculation. It is
enabled only with `MIINFER_Q4K_Q8_1_PACKED_INPUT=1`.

## Environment

```text
GPU:       AMD Instinct MI50 / gfx906
Clocks:    profile_peak (1725 MHz SCLK / 1000 MHz MCLK policy)
Model:     /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Build:     build/mi50-release
Fixture:   /tmp/m6a273-reference-p12
```

## Correctness

The candidate built successfully and passed native 16-token generation and
deterministic replay. Device usage was unchanged at `20,094,914,900` bytes;
decode allocations remained `0`.

## Results

| Workload | Candidate | Control | Candidate delta |
| --- | ---: | ---: | ---: |
| TG64 tok/s | 14.2100 | 14.1591 | +0.36% |
| TG128 tok/s | 13.9439 | 13.9379 | +0.04% |

Each result passed replay. The improvement is below the project's useful
end-to-end threshold and is not repeatably distinguishable from run noise.

## Decision

**REJECT for production.** Keep the current decoded-metadata path as the
default. The candidate remains opt-in diagnostic code only.

## Interpretation

The source-level llama.cpp operand-packing difference does not provide a
measurable whole-token benefit on this MI50 workload when isolated in the
existing MIInfer geometry. Stop this Q4 inner-loop family unless new hardware
profiling identifies a different mechanism.
