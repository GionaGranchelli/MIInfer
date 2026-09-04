# EXP-0092 — M6-A28 native autoregressive GPU generation

## Question

Can the complete Qwen3.8-27B qwen35 GPU executor feed its GPU LM-head argmax
back into the next token while keeping recurrent/KV state and activations on
the GPU, with deterministic replay and no decode-loop allocations?

## Change

Added a Qwen3.8 Q4_K GPU embedding primitive and native generation modes to
the existing complete 64-layer qwen35 GPU executor:

```text
Q4_K embedding on GPU
  → 64 hybrid layers on GPU
  → final norm / Q8_K / Q6_K LM head on GPU
  → GPU first-index argmax
  → selected uint32 token copied to host
```

The candidate uses fixed preallocated layer/state/workspace buffers. It does
not copy logits or recurrent/KV state to the host during the token loop.

## Environment

```text
GPU:       AMD Instinct MI50 / gfx906
model:     /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
fixture:   /tmp/m6a273-reference
clock:     stable_peak
build:     build/mi50-release
```

Commands:

```bash
build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block MODEL FIXTURE --generate16
build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block MODEL FIXTURE --generate64
build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block MODEL FIXTURE --generate128
```

Each mode runs once, resets all recurrent/KV state, runs again, and compares
the generated token IDs and combined logical state fingerprint.

## Results

| Run | Tokens | First token | Last token | Replay | Decode allocations | Device bytes after setup | Peak device bytes |
| ---: | ---: | ---: | ---: | :---: | ---: | ---: | ---: |
| `--generate16` | 16 | 11 | 585 | PASS | 0 | 17,018,706,644 | 17,018,706,644 |
| `--generate64` | 64 | 11 | 29,517 | PASS | 0 | 17,018,706,644 | 17,018,706,644 |
| `--generate128` | 128 | 11 | 15,016 | PASS | 0 | 17,018,706,644 | 17,018,706,644 |

State fingerprints were deterministic for each repeated run:

```text
16:  1853370403272745608
64: 10502567857879740641
128: 3481716422374478905
```

The existing A27 observable retest established finite, highly correlated
final logits through P64. A28's native path completed all requested runs and
GPU argmax feedback without a runtime failure or replay mismatch.

## Decision

**ACCEPT / M6-A28 COMPLETE.**

Native qwen35 autoregressive GPU generation is operational through 128 token
steps. Recurrent state, full-attention KV state, activations, quantized
buffers, and logits remain device-resident during the loop; only the selected
token ID is returned to the host. No production throughput claim is made by
this bring-up experiment.

## Follow-up

Proceed to M6-B0/B1: benchmark the native MI50 generation path against the
pinned upstream llama.cpp baseline under identical model, workload, and clock
conditions. Add a serving-facing API only if the benchmark harness requires
it.
