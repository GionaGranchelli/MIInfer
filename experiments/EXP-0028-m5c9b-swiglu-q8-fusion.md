# EXP-0028 — M5-C9b fused SwiGLU to Q8 quantization

**Status:** CLOSED — REJECTED for production selection
**Milestone:** M5
**Date:** 2026-09-01
**Baseline commit:** `d7a848a`
**Candidate:** opt-in `MIINFER_SWIGLU_Q8_FUSION=fused`

## 1. Question

Can SwiGLU, the production FP16 activation boundary, and Down-input Q8Exact
quantization be fused without changing the quantized payload or the
multi-token decode trajectory?

## 2. Candidate

The control performs three launches per FFN layer:

```text
SwiGLU F32
  -> F32 to FP16
  -> Q8Exact quantization
```

The candidate performs one gfx906 launch that computes the same SiLU product,
round-trips it through FP16 in shared memory, and writes the existing
36-byte `Q8ExactBlock` stream directly. The Down GEMV, its output conversion,
and all other kernels are unchanged. The candidate is selected only for the
trace-free fast path; the default remains the control path.

The interleaved A/B harness mode is:

```bash
build/mi50-release/miinfer-qwen3-attention-ab-bench \
  /path/to/Qwen3-8B-q4_0-b968826d.gguf \
  --mode swiglu-fusion --warmup 8 --generated-tokens 64 --pairs 3
```

## 3. Correctness results

The direct GPU test compared the separate and fused Q8Exact blocks over the
full 12,288-element intermediate vector (384 blocks), using deterministic
values spanning approximately `[-1000, 1000]`. The payload and metadata were
byte-identical: **PASS**.

The existing Release CTest suite remained green with the default path:

```text
19/19 tests passed
```

The fused path also passed the focused GPU tests and the existing short
eight-token decode sequence. However, the fixed-prefix audit across the full
64-token workload found the first token divergence at position 38:

| Position | Separate | Fused |
|---:|---:|---:|
| 38 | 9370 | 104796 |

The two paths matched through position 37. Since greedy decoding feeds the
selected token back into the next step, later IDs are no longer a controlled
comparison after this point. The candidate therefore fails the required
long-generation behavioral gate despite passing the standalone payload test.

## 4. Performance results

The standalone microbenchmark used the Qwen3 intermediate shape (`12288`
elements, 30 warmups, 1000 HIP-event iterations):

| Path | Median | Mean | Launches |
|---|---:|---:|---:|
| Separate | 12.800 µs | 13.385 µs | 3 |
| Fused | 8.480 µs | 8.608 µs | 1 |

The microbenchmark reports a 33.8% local speedup, but this does not overcome
the full-model correctness failure. The 8-token interleaved end-to-end A/B
also produced identical IDs and measured 59.6674 tok/s separate versus
60.3632 tok/s fused (+1.17%).

The independent 64-token trace-free runs measured 54.450 tok/s separate and
53.918 tok/s fused. Because their histories diverged at position 38, this is
not a valid long-context performance comparison and is recorded only as
supporting evidence; it is not a speedup claim.

The candidate's structural profile at P64 was:

```text
total dispatches: 1625 -> 1553
SwiGLU + Down-input quantization stages: 108 -> 36 launches
```

The reduction is 72 launches, not 54, because the current control includes
both the F32-to-FP16 conversion and the Q8Exact quantization launch after
SwiGLU.

## 5. Interpretation

The fusion mechanism is locally faster and the synthetic payload test is
strong, but that evidence is insufficient for this workload. The actual
model decode trajectory changes after 38 positions, so the fused candidate
cannot be production-selected without first explaining the model-input case
where its Q8 stream differs from the separate path or identifying another
state interaction.

This is a correctness rejection, not evidence that the microbenchmark or the
fusion idea is intrinsically useless. The separate Gate/Up quantization reuse
candidate remains cleaner to investigate because it avoids changing the
SwiGLU-to-Q8 arithmetic boundary and has an independently measured target.

## 6. Decision

```text
REJECT — retain separate production path
```

Keep the fused kernel and A/B harness as opt-in diagnostic code only. Do not
widen tolerances or alter production precision semantics.

## 7. Follow-up

Investigate the actual-model Q8 mismatch only if a byte-level validation path
can be added cheaply. Otherwise move to the separate Gate/Up quantization
reuse experiment, with the current control as the baseline.
