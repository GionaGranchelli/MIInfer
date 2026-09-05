# EXP-0161 — M6-B65 post-B62 native benchmark baseline

## Question

What is the reproducible TG64 baseline for the current accepted Qwen3.8-27B
GPU path after the B62 rejection?

## Environment

```text
GPU:       AMD Instinct MI50 / gfx906
Clocks:    profile_peak / stable peak
Model:     /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Build:     build/mi50-release
Fixture:   /tmp/m6a273-reference
Workload:  --bench64
Samples:   5
```

## Command

```bash
build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  /tmp/m6a273-reference --bench64
```

## Results

```text
samples_ms: 4510.29, 4512.44, 4516.52, 4517.73, 4519.86
median_ms:  4516.52
median_tok_s: 14.1702
replay: PASS
allocations_during_decode: 0
device_bytes_after_setup: 20094914900
peak_device_bytes: 20094914900
```

## Decision

**MEASUREMENT-ONLY / BASELINE.** The accepted path is deterministic and
allocation-free. FFN Down and recurrent-layer execution remain the measured
optimization targets; no candidate was changed by this experiment.

## Follow-up

Use this same-build baseline for one new, evidence-backed recurrent-layer or
FFN Down experiment. Do not repeat rejected geometry, staging, or row-wave
variants without a materially different mechanism.
