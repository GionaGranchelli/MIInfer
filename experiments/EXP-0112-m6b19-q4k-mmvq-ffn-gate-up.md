# EXP-0112 — M6-B19 Q4_K×Q8_1 MMVQ FFN Gate/Up

## Question

Can the Q4_K FFN Gate and Up projections share one Q8_1 activation and use
the existing gfx906 Q4_K×Q8_1 MMVQ primitive to reduce whole-token decode
time?

## Hypothesis

After B18, FFN Gate/Up remains the largest repeated recurrent-layer stage.
The two projections consume the same normalized hidden vector, so one Q8_1
quantization can feed both projections without changing their required
matrix-vector work.

## Candidate and control

The control is the B18 production path selected with
`MIINFER_Q4K_Q8_1_MMVQ_FFN_GATE_UP=0`: Q4_K×Q8_K Gate/Up projections with
the existing shared Q8_K input reuse. The candidate uses one Q8_1 quantization
of the normalized FFN input, then two independent 128-thread/output-row
Q4_K×Q8_1 MMVQ projections. FFN Down remains on the B18-selected Q8_1 MMVQ
path. No kernel geometry, activation, or state-update changes are included.

## Environment

```text
GPU: AMD Instinct MI50 / gfx906
Clock policy: stable_peak (1725 MHz SCLK / 1000 MHz MCLK observed)
Model: /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Build: build/mi50-release
ROCm: 7.1.52802-9999
Compiler: Clang 20.0.0.rocm
Fixture: /tmp/m6a273-reference-p12
```

## Correctness and resource checks

* Candidate native 16-token generation replay passed.
* Candidate 64-layer external observable-contract run completed successfully;
  logits retained high cosine and matching top-k behavior under the accepted
  M6 external contract.
* Release CTest: `20/20 PASS`.
* Candidate and control used `17019965780` post-setup device bytes and zero
  decode-loop allocations.

## Results

Three serial process medians, each process containing five timed samples:

| Workload | Control median | Candidate median | Change |
| --- | ---: | ---: | ---: |
| TG64 tok/s | 10.1707 | 10.7750 | +5.94% |
| TG128 tok/s | 10.0231 | 10.6477 | +6.23% |

The corresponding median times were 6292.59 ms versus 5939.66 ms for TG64,
and 12770.60 ms versus 12021.30 ms for TG128. The candidate profile showed
`ffn_gate_up=0.44624 ms` versus `0.48896 ms` for control in the representative
layer-0 sample; whole-token measurements are the selection authority.

## Interpretation

The candidate removes the duplicated Q8_K Gate/Up preparation and replaces
both projections with the already validated Q4_K×Q8_1 MMVQ mechanism. The
gain is repeatable at both tested generation lengths and exceeds the useful
whole-token threshold without increasing peak device allocation.

## Decision

**KEEP; production-selected.** The shared Q4_K×Q8_1 Gate/Up path is now the
default. Set `MIINFER_Q4K_Q8_1_MMVQ_FFN_GATE_UP=0` to run the B18 control.

## Follow-up

Refresh the complete post-B19 profile before choosing another optimization.
Do not assume remaining dispatch count is the next bottleneck.
