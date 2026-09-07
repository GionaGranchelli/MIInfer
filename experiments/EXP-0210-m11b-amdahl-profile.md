# EXP-0210 — M11-B Amdahl Profile and Sequential-Floor Check

## Hypothesis

The remaining gap after B=4 layer-major projection reuse might be caused by
the recurrent state transition, requiring a block scan before more projection
work can matter.

## Measurement

Used the existing `miinfer-m6a21-qwen35-gpu-hybrid-block --profile64` HIP-event
profiler at the validated position 63 on gfx906 with the qualification model.
The profiler reported 61.856 ms total GPU time for the token-major step and
profiled representative recurrent layer 0 and full-attention layer 3.

Representative recurrent-layer stage timings:

| Stage | GPU ms | Interpretation |
| --- | ---: | --- |
| QKV projection | 0.204 | batchable projection |
| gate projection | 0.080 | batchable projection |
| beta/alpha | 0.044 | token-local |
| conv/head norm | 0.050 | token-local |
| recurrent state update | 0.061 | causally sequential |
| recurrent gate | 0.010 | token-local |
| SSM output projection | 0.096 | batchable projection |
| FFN gate/up | 0.175 | batchable projection |
| FFN down | 0.462 | batchable projection |
| remaining residual/norm/activation | 0.077 | token-local/fused |

The recurrent state update is approximately 4--5% of the representative
recurrent layer event. Projection and FFN work dominate the measured layer
cost. The full-attention layer's largest stages were FFN down (0.468 ms),
FFN gate/up (0.174 ms), Q projection (0.098 ms), V projection (0.064 ms), and
cached attention (0.056 ms).

## Amdahl implication

The profile does not justify a recurrent scan as the next experiment: even an
ideal state-update removal cannot provide the required 2.5x production gain.
The next credible path is a production-shape Q4_K/Q6_K skinny-GEMM mapping
that improves the dominant 5120/17408-column projections beyond the validated
B=4 GEMV reuse.

## Decision

**RETEST projection mapping; defer recurrent scan.** The current B=4
layer-major result remains the qualified candidate at 39.96 P512 tok/s. The
100 tok/s gate is not met, and this profile establishes the next bottleneck
without relaxing that gate.
