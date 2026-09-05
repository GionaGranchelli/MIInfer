# EXP-0136 — M6-B44 Q4_K×Q8_1 split-K MMVQ mapping

## Question

Does the pinned llama.cpp gfx906 MMVQ mapping—two Wave64s cooperating on one
Q4_K output row—outperform MIInfer's current two-row workgroup mapping?

## Candidate

The candidate used two Wave64s to split each output row's K dimension, with the
existing Q4_K×Q8_1 dot4 arithmetic, decoded metadata, LDS activation staging,
and final FP32 reduction contract unchanged. It was opt-in through
`MIINFER_Q4K_Q8_1_SPLIT_K=1` and was never production-selected.

## Environment

```text
GPU:       AMD Instinct MI50 / gfx906
Clock:     stable_peak (1725 MHz SCLK / 1000 MHz MCLK observed)
Model:     /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Build:     build/mi50-release
Fixture:   /tmp/m6a273-reference-p12
```

## Correctness

Native 16-token generation and both benchmark replays passed. Decode-loop
allocations remained `0`, and device usage remained `17,019,965,780` bytes.

## Results

Serial same-build medians, candidate versus B41 control:

| Workload | Control tok/s | Candidate tok/s | Change |
| --- | ---: | ---: | ---: |
| TG64 | 13.8779 | 11.6227 | -16.25% |
| TG128 | 13.6825 | 11.4662 | -16.20% |

## Interpretation

The external mapping is not a good fit for this MIInfer Q4_K×Q8_1 workload.
Splitting K across two waves on one row loses about 16% end-to-end despite
preserving correctness. The current two independent output-row mapping remains
the better choice on MI50.

## Decision

**REJECT.** Remove the candidate and retain B41's production mapping. Do not
repeat this split-K geometry without new hardware-level evidence.

## Follow-up

The next experiment must target a different measured cost; Q4_K workgroup
mapping is closed for the current path.
