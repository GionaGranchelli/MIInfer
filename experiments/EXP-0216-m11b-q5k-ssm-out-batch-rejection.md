# EXP-0216 — M11-B Recurrent Q5_K `ssm_out` B=4 Projection

## Hypothesis

The layer-major path batches QKV and FFN projections but still runs the native
Q5_K recurrent `ssm_out` projection once per token. Reusing the existing Q5_K
B=4 kernel for the four deferred recurrent outputs should reduce the remaining
prefill cost.

## Baseline

The validated opt-in layer-major schedule with native Q5_K `ssm_out` kept its
existing four scalar projection launches per recurrent layer.

## Candidate

Quantize the four deferred recurrent gated outputs into the existing B=4 Q8_1
workspace and call `launch_q5k_wave_gemv_batched4` for the `[6144 -> 5120]`
native Q5_K projection. The candidate was temporary and was removed after the
measurement.

## Environment

- GPU: AMD Instinct MI60/MI50-visible gfx906
- SCLK/MCLK: 1606/1000 MHz
- ROCm: 6.4.0 / LLVM 20
- Build: `mi50-release`
- Model: `Qwen3.8-27B-Q4_K_M.gguf`
- Prompt: 513 repeated `hello` tokens
- `MIINFER_PREFILL_LAYER_MAJOR=1`
- `MIINFER_HIP_GRAPH=0`
- Native QKV, V, attention gate, Q5_K `ssm_out`, Q4_K gate/up/down enabled

## Correctness

The candidate and scalar control produced the same 16-token `hello`
continuation at P16. The P513 measurements were prefill-only; no long
generation qualification was claimed.

## Results

Three release repetitions per side on P513:

| Path | Runs (ms) | Median | Throughput |
| --- | --- | ---: | ---: |
| Scalar native `ssm_out` control | 12839.38 / 12831.12 / 12825.50 | 12831.12 ms | 39.98 tok/s |
| Q5_K B=4 `ssm_out` candidate | 16230.47 / 16240.67 / 16230.07 | 16230.47 ms | 31.61 tok/s |

The candidate regressed production-shaped prefill by approximately 20.9%.
The shorter P16 check also regressed from 40.60 to 32.02 tok/s.

## Interpretation

The existing generic Q5_K B=4 kernel does not provide the same reuse benefit
as the accepted Q4_K projection path. Its multi-output decoder state and
register/work scheduling cost outweigh the saved dispatches for this shape.

## Decision

**REJECT.** Restore scalar native Q5_K `ssm_out` launches. No runtime switch or
additional workspace remains.

## Follow-up

Any Q5_K `ssm_out` optimization needs a shape-specific kernel mapping or a
profiled reduction in decoder/register overhead; reusing the generic B=4
template is not sufficient.
