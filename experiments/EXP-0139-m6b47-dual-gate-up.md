# EXP-0139 — M6-B47 dual Gate/Up Q4_K×Q8_1 projection

## Question

Can the Gate and Up Q4_K×Q8_1 projections share one staged Q8_1 input in a
single opt-in kernel without changing their validated arithmetic or whole-token
behavior?

## Baseline and candidate

The control uses the production B41 Q4_K×Q8_1 path: one Q8_1 activation is
already quantized, then Gate and Up run as separate projections with decoded
Q4_K metadata staged in LDS. The B47 candidate used one 256-thread workgroup
per output row, computed one Gate row and one Up row, and staged the shared
Q8_1 input once per workgroup. Weight payload access, decoded metadata, dot4
arithmetic, and per-row reductions were retained.

The candidate was opt-in through `MIINFER_Q4K_Q8_1_DUAL_GATE_UP=1`; production
defaults were unchanged.

## Environment

```text
GPU: AMD Instinct MI50 / gfx906
Clock: stable_peak
Model: /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Build: build/mi50-release
Correctness fixture: /tmp/m6a273-reference-p12
```

## Correctness

The candidate passed:

* native 16-token generation and deterministic replay;
* 64-layer external observable contract;
* poisoned reset/replay;
* Release CTest: 20/20;
* zero decode-loop allocations;
* unchanged device usage in the generation run.

## Results

Serial same-build medians:

| Workload | Control tok/s | Candidate tok/s | Change |
| --- | ---: | ---: | ---: |
| TG64 | 13.9191 | 13.8895 | -0.21% |
| TG128 | 13.6887 | 13.6838 | -0.04% |

The candidate reduced the two Gate/Up projection launches to one combined
launch, but the shared staging did not produce a repeatable whole-token gain.
The small regressions are below the useful threshold and consistent with
measurement noise/extra combined-kernel overhead.

## Decision

**REJECT.** The dual Gate/Up projection kernel is removed. The B41 production
path remains selected.

## Follow-up

Do not try another Gate/Up fusion or geometry variant without new profiling
evidence. The next experiment must target a different measured cost.
