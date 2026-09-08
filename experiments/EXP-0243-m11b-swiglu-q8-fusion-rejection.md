# EXP-0243 — M11-B SwiGLU/Q8 producer-consumer fusion rejection

## Hypothesis

Fusing the paired Q4_K gate/up SwiGLU producer with Q8_1 quantization would
remove the FP32 activation write/read and four standalone quantizer launches
from the recurrent FFN tail.

## Baseline and candidate

- Baseline: existing B=4 paired Q4_K gate/up SwiGLU to FP32 activation,
  followed by one Q8_1 quantizer launch per token.
- Candidate: one 512-thread workgroup computes 32 FFN rows, stages four token
  activations in LDS, and emits Q8_1 blocks directly.

## Environment

- GPU: AMD Instinct MI50, gfx906
- Model: `Qwen3.8-27B-Q4_K_M.gguf`
- Shape: Q4_K gate/up `[17408, 5120]`, four tokens
- Build: `mi50-release`
- Benchmark: interleaved 41-round HIP-event timing after 20 warmup pairs

## Correctness

The candidate Q8_1 output was byte-identical to the baseline output.

## Results

| Path | Median | Relative |
| --- | ---: | ---: |
| Existing SwiGLU + four Q8 launches | 372.959 us | 1.000x |
| Fused SwiGLU/Q8 candidate | 475.999 us | 0.784x |

## Interpretation

The candidate reduced dispatch count, but the 512-thread workgroup and four
rows of per-wave accumulation increased register/workgroup cost enough to
lose 21.6%. The removed global activation traffic was not the bottleneck.

## Decision

**REJECT.** The candidate and benchmark were removed; the production path is
unchanged.

## Follow-up

Do not fuse this boundary without a materially different mapping that avoids
the extra register/workgroup cost. The measured prefill gate remains open.
