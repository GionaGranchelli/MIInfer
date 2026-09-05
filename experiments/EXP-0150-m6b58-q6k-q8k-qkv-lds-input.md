# EXP-0150 — M6-B58 Q6_K×Q8_K QKV LDS input

## Question

Can recurrent QKV projection stage its shared Q8_K activation in LDS while
two independent output rows reuse it?

## Candidate

The opt-in candidate used two 64-lane reductions in a 128-thread workgroup and
staged the Q8_K activation blocks once per workgroup. Q6_K decoding, dot4
arithmetic, accumulation, and output contract were unchanged. The default
production path remained the existing one-row kernel.

## Environment

```text
GPU:       AMD Instinct MI50 / gfx906
Model:     /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Build:     build/mi50-release
Clock:     stable_peak; 1725 MHz SCLK / 1000 MHz MCLK observed
Fixture:   /tmp/m6a273-reference-p12
```

## Correctness

Native 64-token replay passed with identical last token and state fingerprint.
Decode allocations remained zero and device bytes remained
`17,019,965,780`.

## Results

Same-build medians (control values are the B57 control):

| Workload | Control tok/s | Candidate tok/s | Change |
| --- | ---: | ---: | ---: |
| TG64 | 14.1004 | 14.0241 | -0.54% |
| TG128 | 13.8573 | 13.7884 | -0.50% |

The candidate profile increased representative recurrent QKV projection from
about `0.173–0.184 ms` to `0.187–0.202 ms`.

## Interpretation

The repeated Q8_K activation reads are not costly enough to repay LDS staging
and the two-row workgroup overhead for this QKV shape.

## Decision

**REJECT.** The candidate was removed; the existing Q6_K×Q8_K dot4 QKV path
remains production-selected.

## Follow-up

Do not repeat shared-input staging for this QKV kernel without a materially
different access or occupancy hypothesis.
