# EXP-0138 — M6-B46 Q4_K×Q8_1 packed-input diagnostic

## Question

Does explicitly packing each lane's Q8 operand words and scales into local
values improve the B41 Q4_K×Q8_1 inner loop on gfx906, while retaining B41's
two-row workgroup mapping and decoded metadata staging?

## Baseline and candidate

The B41 control uses a 256-thread workgroup with two independent 128-thread
output rows, LDS-resident Q8_1 input, decoded Q4_K metadata in LDS, global Q4
payload loads, the existing dot4 arithmetic, and the existing reduction.

The opt-in B46 candidate changed only the inner helper's operand form: it
loaded the two Q8 word pairs and two Q8 scales into local arrays before the
existing two-part dot loop. Launch geometry, Q4 payload access, decoded
metadata, arithmetic, and output reduction were unchanged.

## Environment

```text
GPU: AMD Instinct MI50 / gfx906
Clock: stable_peak
Model: /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Build: build/mi50-release
Baseline: B41 production path
```

## Correctness

The candidate passed:

* native 16-token generation and deterministic replay;
* the 64-layer external observable contract;
* poisoned reset/replay;
* Release CTest: 20/20;
* zero decode-loop allocations;
* unchanged device usage: `17,019,965,780` bytes.

## Results

Serial same-build medians, candidate versus control:

| Workload | Control tok/s | Candidate tok/s | Change |
| --- | ---: | ---: | ---: |
| TG64 | 13.8938 | 13.8953 | +0.01% |
| TG128 | 13.6719 | 13.6822 | +0.08% |

The differences are below the useful threshold and within run dispersion.

## Decision

**REJECT.** Explicit local Q8 operand packing does not produce a repeatable
whole-token gain on the current MI50 workload. The diagnostic code was removed;
B41 remains the production path. This result does not justify more Q4 geometry,
split-K, or operand-packing variants without new profiling evidence.

## Follow-up

The next experiment must target a different measured cost or obtain genuine
hardware-level resource evidence. The current accepted benchmark remains about
13.9 TG64 and 13.7 TG128 tok/s versus the pinned llama.cpp control at about
22.5 and 22.3 tok/s.
