# EXP-0158 — M6-B66 expanded Q4_K FFN Gate/Up

## Question

Does reusing the existing expanded-Q4_K representation for FFN Gate and Up
improve the repeated projection cost without changing model behavior?

## Candidate

The opt-in `MIINFER_Q4K_EXPANDED_GATE_UP=1` path used the existing
model-load Q4_K expansion and expanded Q4_K×Q8_1 kernel already selected for
FFN Down. No new geometry or quantization was introduced.

## Environment

```text
GPU:       AMD Instinct MI50 / gfx906
Model:     /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Build:     build/mi50-release
Fixture:   /tmp/m6a273-reference-p12
Clocks:    profile_peak
```

## Correctness

Native generation replay alone reported `PASS`, but the external 64-layer
contract failed immediately at position 0, layer 0:

```text
max_abs:       15.1249
RMS:            0.216807
relative RMS:   0.812299
```

The candidate was not performance-tested. Production defaults were restored.

## Decision

**REJECT.** The existing expanded Down representation/kernel is not
interchangeable with the Gate/Up production contract. Do not select this path
without first resolving the representation discrepancy.
