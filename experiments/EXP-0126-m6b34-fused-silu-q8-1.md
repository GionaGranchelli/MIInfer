# EXP-0126 — M6-B34 fused SiLU to Q8_1

## Question

Can the recurrent and full-attention FFN Down paths avoid materializing the
FP32 SiLU result by producing the existing FP16-rounded Q8_1 input directly?

## Hypothesis

The current Q4_K×Q8_1 Down path launches SiLU/multiply, then quantizes its
result in a separate operation. A candidate preserving the FP16 rounding
boundary could remove that intermediate materialization and one launch per
FFN path.

## Baseline and candidate

The baseline is the B32 production path at commit `6ee887e`: separate SiLU,
then Q8_1 quantization, then the existing Q4_K×Q8_1 Down GEMV. The candidate
used an opt-in `MIINFER_Q4K_Q8_1_FUSED_SILU=1` path for Q4_K Down projections
in recurrent and full-attention layers. The candidate explicitly rounded the
SiLU result to FP16 before Q8_1 serialization. It was removed after testing.

## Environment

```text
GPU:       AMD Instinct MI50 / gfx906
Clock:     stable_peak (1725 MHz SCLK / 1000 MHz MCLK observed)
Model:     /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
Build:     build/mi50-release
Fixture:   /tmp/m6a273-reference-p12
Baseline:  6ee887e
```

## Correctness

Native 16-token generation passed deterministic replay with zero decode
allocations and unchanged `17,019,965,780` tracked/peak device bytes. The
complete P64 external observable contract passed without changing tolerances;
teacher-forced decisions matched through the tested positions, P64 argmax was
`8719`, and P64 logits cosine was `0.999556` with 5/5 top-5 overlap. The
candidate's native sequence differed from the old MIInfer sequence, which is
permitted only because M6 uses the pinned external observable contract.

Release CTest after restoring the production path passed `20/20`.

## Benchmark

Five-sample medians per process at stable_peak:

| Workload | B32 control tok/s | Candidate tok/s | Change |
| --- | ---: | ---: | ---: |
| TG64 | 12.0637 | 12.0741 | +0.09% |
| TG128 | 11.8983 | 11.9075 | +0.08% |

The result is below the project's useful `0.5%` threshold and within normal
run variation. It did not provide a demonstrated whole-token benefit despite
removing the separate SiLU launch/materialization boundary.

## Interpretation

Preserving the FP16 boundary was sufficient for the tested external
observable contract, but this fusion does not improve production throughput on
the MI50. The necessary quantization and Down GEMV work dominate the small
support-kernel savings.

## Decision

**REJECT for production.** The opt-in kernel, launcher, and selection logic
were removed. B32's transposed no-decay-store path remains the production
default.

## Follow-up

Do not revisit this fusion without a materially different implementation that
reduces measured device work rather than only launch count.
