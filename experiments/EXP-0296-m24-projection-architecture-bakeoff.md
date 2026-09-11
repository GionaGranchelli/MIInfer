# EXP-0296 — M24 resident MMQ versus GPU dequantized FP16 GEMM

## Hypothesis

At B512, resident quantized MMQ may be slower than expanding one projection
to reusable FP16 scratch on the GPU and using rocBLAS GEMM. The bake-off must
include the GPU dequantization cost, but not host transfers or model loading.

## Candidate

Compare the current resident M23 Q4_K MMQ path with:

```text
resident canonical Q4_K → GPU Q4_K→FP16 expansion → rocBLAS FP16 GEMM
```

The benchmark uses the exact `blk.8.ffn_down.weight` shape
`rows=5120, columns=17408` and measures B64/B128/B256/B512.

## Environment

- AMD Instinct MI50 / gfx906
- Qwen3.8-27B-Q4_K_M
- release build, ROCm 6.4.0
- `miinfer-m12-dense-stage-bench`
- synthetic FP16 input with deterministic values

The device was in `Performance Level: auto` with the low-power warning. The
observed idle state was `mclk=350 MHz`, `sclk=930 MHz`; setting high
performance requires sudo unavailable to this session. These measurements are
therefore not valid end-to-end performance claims.

## Correctness

- GPU Q4_K→FP16 expansion: `max_repack_error=1.52588e-05`.
- Resident MMQ output was finite.
- The direct MMQ-versus-FP16-GEMM maximum absolute difference was `6.5378`.
  This is not a model-level parity result: the paths use different activation
  precision and accumulation behavior. Model logits and continuation must be
  checked before adopting FP16 prefill.

## Results

Median kernel time in microseconds; `repack+GEMM` includes one GPU expansion.

| batch | resident MMQ | FP16 GEMM | repack + GEMM | MMQ / FP16 path |
|---:|---:|---:|---:|---:|
| 64 | 4,910.23 | 2,253.76 | 3,439.04 | 1.43× |
| 128 | 6,354.71 | 2,233.60 | 3,418.88 | 1.86× |
| 256 | 11,420.31 | 3,499.52 | 4,684.80 | 2.44× |
| 512 | 18,790.06 | 5,313.91 | 6,499.19 | 2.89× |

The quantized resident tile occupies `72,417,280` bytes; one FP16 expansion
occupies `178,257,920` bytes.

## Interpretation

The B512 result supports the architectural hypothesis: the current resident
MMQ primitive is not automatically the best P512 execution model. The result
is especially strong because the FP16 candidate includes GPU expansion and
the MMQ candidate uses the already-resident packed tile.

This does not yet establish an end-to-end win. Clock state is contaminated,
the benchmark covers one Q4 projection, and the model's Q4 Gate/Up and Q6/Q4
Down shapes need separate coverage.

## Decision

**RETEST.** Keep the current resident-all path unchanged. Treat GPU
dequantized FP16 GEMM as the leading M24 prefill candidate, pending valid
clock-captured runs and model-level correctness.

## Follow-up

1. Repeat with fixed clocks and capture SCLK/MCLK/power during the run.
2. Add the transposed Q4 Gate/Up shape and Q6/Q4 Down cases.
3. Wire one recurrent layer through FP16 scratch and compare intermediate
   tensors, logits, and continuation output against resident MMQ.
4. If parity passes, measure the complete P512 path before any fusion work.
