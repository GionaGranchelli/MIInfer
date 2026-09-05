# EXP-0142 — M6-B50 fused recurrent output path

## Question

Can the transposed no-decay-store DeltaNet update fuse head normalization and
the recurrent output gate, matching the higher-level mechanism used by
llama.cpp's fused Gated Delta Net path?

## Candidate

An opt-in gfx906 kernel retained the existing recurrence, state layout, FP32
arithmetic, and 128-thread head mapping. It computed recurrent output, head
RMS normalization, SSM normalization, and sigmoid gating in one dispatch,
writing directly to the recurrent projection input. Diagnostic capture paths
continued to use the production implementation.

## Environment

```text
GPU:       AMD Instinct MI50 / gfx906
Clock:     stable_peak (1725 MHz SCLK / 1000 MHz MCLK)
Model:     /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Build:     build/mi50-release
Fixture:   /tmp/m6a273-reference-p12
Baseline:  B41 production path
```

## Correctness

Native 16-token generation passed exact replay with first token `11`, last
token `585`, zero decode allocations, and unchanged tracked device usage of
`17,019,965,780` bytes.

## Results

Fresh same-build serial medians:

| Workload | Control tok/s | Candidate tok/s | Change |
| --- | ---: | ---: | ---: |
| TG64 | 13.8821 | 13.9265 | -0.32% |
| TG128 | 13.6559 | 13.6943 | -0.28% |

The candidate was slower on both workloads; native replay passed in both
candidate runs.

## Decision

**REJECT.** The larger recurrent output fusion does not improve whole-token
decode on gfx906. The candidate was removed and the existing transposed
no-decay-store path remains production-selected.

## Follow-up

The llama.cpp fused Gated Delta Net mechanism is a useful architectural
reference, but this direct post-update fusion is not a viable MIInfer kernel
mapping. Do not repeat it without a materially different state/projection
execution design.
