# EXP-0094 — M6-B2 direct layer-output handoff

## Question

Are the 64 blocking layer-output device-to-device copies in the native
Qwen3.8 decode loop a material source of the current MI50 performance gap?

## Hypothesis

Writing the final residual directly into the caller-provided next-layer buffer
may remove one synchronous device-to-device boundary per layer and improve
steady-state generation throughput.

## Baseline

Baseline commit: `0f6d39265cc9`.

The production-shaped path writes each layer residual to its private
`layer_output` buffer, then performs a blocking `hipMemcpy` into the next
layer's output buffer. The accepted M6-B1 TG64 result was approximately
3.37 tok/s.

## Candidate

The transient candidate changed only the final residual destination in the
recurrent and full-attention layer executors from `layer_output` to the
caller-provided `output`, removing the following copy from both paths:

```text
final residual → layer_output → blocking D2D copy → output
```

No arithmetic, state update, cache, quantization, kernel, or scheduling logic
was changed. The candidate was tested in the working tree and then reverted;
it was not production-selected.

## Environment and workload

```text
GPU:          AMD Instinct MI50 / gfx906
Model:        /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Clock policy: stable_peak
Build:        build/mi50-release
Fixture:      /tmp/m6a273-reference
```

Commands:

```bash
build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  /tmp/m6a273-reference --generate16
build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  /tmp/m6a273-reference --generate64
```

## Results

| Workload | Baseline tok/s | Candidate tok/s | Change |
| --- | ---: | ---: | ---: |
| TG16, first pass | 3.3855 | 3.3864 | +0.03% |
| TG16, replay pass | 3.3835 | 3.3850 | +0.04% |
| TG64, first pass | 3.3755 | 3.3743 | −0.04% |
| TG64, replay pass | 3.3719 | 3.3687 | −0.10% |

Correctness remained intact: TG16 and TG64 token sequences and state
fingerprints replayed identically, decode allocations remained zero, and the
tracked device footprint remained `17,018,706,644` bytes.

## Interpretation

The 64 layer handoff copies are not a material bottleneck in this native
Qwen3.8 path. Removing them produced no repeatable end-to-end improvement;
the observed differences are below measurement noise. This differs from the
M5 result only in workload context: the Qwen3.8 executor's dominant cost is
elsewhere, despite the copies being structurally unnecessary.

## Decision

**REJECT.** Restore and retain the existing layer-output handoff. Do not
revisit this copy family without new profiling evidence.

## Follow-up

Build a production-shaped per-family profile for one warm steady-state decode
token, covering recurrent projections/state update, full-attention work,
FFN, normalization, quantization/conversion, LM head, synchronization, and
unaccounted wall time. Select the next optimization from the largest measured
whole-token cost.
