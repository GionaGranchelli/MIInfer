# EXP-0278 — M22 quantized batch-width study

## Hypothesis

Replacing repeated B4 quantized GEMV launches with the existing M13 B8
quantized matrix-style kernel will improve the M12 deferred tail.

## Baseline and candidate

The baseline is the production B4 reuse kernel. The candidate is the existing
M13 B8 batched kernel, measured on exact model projection shapes. This is a
lab-only test; the runtime was not changed.

## Environment and benchmark

* Model: Qwen3.8-27B-Q4_K_M.gguf
* SHA-256: `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
* GPU: MI50/gfx906, SCLK 1606 MHz, MCLK 1000 MHz
* Raw result: `results/m22-m13-quant-mm/20260909.json`
* Command: `build/mi50-release/miinfer-m13-quant-mm-bench MODEL.gguf`
* Batch curve: 64/128/256/512/2048, seven timed samples with median reported

## Results

| shape | B64 baseline/candidate us | candidate speed vs B4 | max error |
|---|---:|---:|---:|
| Q4_K FFN-down | 5,426.90 / 6,305.62 | 0.861x | 0.000000 |
| Q6_K QKV | 2,316.33 / 3,326.57 | 0.696x | 0.000000 |

The candidate remained slower across all tested widths. Q4_K candidate speed
was 0.861/0.862/0.863/0.862/0.861x at B64/B128/B256/B512/B2048. Q6_K was
0.696/0.695/0.717/0.718/0.719x. Numerical output matched the B4 reference
within the reported zero maximum absolute difference.

## Interpretation

The current B8 mapping saves launches but increases per-wave live state and
does not exploit the MI50 efficiently. It cannot explain or close the
approximately 4x MIInfer-vs-mx PP gap in its current form.

## Decision

REJECT promoting the existing M13 B8 kernel. Preserve the B4 production path
and continue with a genuinely matrix-oriented quantized kernel or a measured
reference-layout-compatible alternative.

## Follow-up

Profile the reference's Q4_K/Q6_K MM mapping and implement only the first
dominant family after its correctness contract is isolated.
