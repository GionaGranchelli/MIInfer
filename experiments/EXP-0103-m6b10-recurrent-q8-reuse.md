# EXP-0103 — M6-B10 recurrent Q8_K input reuse

## Question

Does each recurrent layer quantize the same normalized activation twice for
the combined QKV projection and the attention-gate projection, and does
reusing that Q8_K buffer improve end-to-end decode?

## Candidate

An opt-in path quantized the recurrent normalized input for QKV once and
allowed the following gate projection to consume the same buffer. The Q8_K
format, projection kernels, ordering, and state updates were unchanged.

## Environment

```text
GPU:          AMD Instinct MI50 / gfx906
Clock policy: stable_peak (1725 MHz SCLK / 1000 MHz MCLK observed)
Model:        /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Build:        build/mi50-release
ROCm:         7.1.52802-9999
Fixture:      /tmp/m6a273-reference
```

## Results

The immediately adjacent control and candidate runs were:

| Workload | Control | Shared | Delta |
| --- | ---: | ---: | ---: |
| TG64 | 7.68412 tok/s | 7.68965 tok/s | +0.07% |
| TG128 | 7.55946 tok/s | 7.56106 tok/s | +0.02% |

Raw samples:

```text
Control TG64: 8314.6, 8323.89, 8328.86, 8339.06, 8340.55 ms
Candidate TG64: 8317.02, 8320.64, 8322.87, 8337.05, 8343.85 ms
Control TG128: 16826.6, 16901.2, 16932.4, 16946.5, 16957.5 ms
Candidate TG128: 16830.6, 16903.2, 16928.8, 16942.5, 16944.2 ms
```

Both paths passed native replay, used zero decode-loop allocations, and held
device usage at `17,018,706,644` bytes.

## Decision

**REJECT.** The duplicated quantization is not a meaningful end-to-end cost
for this workload; both improvements are below the 0.5% useful threshold.

The candidate was reverted and the separate production path remains active.

## Follow-up

Do not spend another milestone on small Q8_K reuse variants without evidence
that quantization has become a measurable whole-token cost. The next target
should address the large recurrent Q4 projection or recurrent state-update
families with a mechanism capable of materially reducing their cost.
