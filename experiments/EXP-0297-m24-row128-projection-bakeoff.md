# EXP-0297 — M24 exact resident-MMQ projection bake-off

## Hypothesis

The actual M23 resident MMQ representation can feed a reusable FP16 scratch
buffer directly, and rocBLAS can beat the production row-128 MMQ kernels at
P512 across the Q4/Q5/Q6 projection mix without retaining a second canonical
GPU weight copy.

## Baseline and candidate

- Baseline: resident M23 repacked MMQ tile plus Q8 activation and MMQ.
- Candidate: the same resident tile, direct GPU expansion to row-major FP16,
  then FP16 rocBLAS GEMM. `resident_fp16_total_us` includes expansion.
- Decode architecture is unchanged; this is a prefill projection experiment.

## Environment

- AMD Instinct MI50 / gfx906, Qwen3.8-27B-Q4_K_M
- ROCm 6.4.0, release build
- `MIINFER_M23_REPACKED_ROW128=1`
- user-selected high-performance state: SCLK 1725 MHz, MCLK 1000 MHz
- batches B64/B128/B256/B512; synthetic deterministic FP16 input

## Correctness

The direct resident expansion is compared against the canonical GPU
dequantizer on sampled rows. Q4, Q5, and Q6 resident expansion also pass the
full synthetic fixture in `qwen35_conv_batch_test`. The benchmark rejects
non-finite resident MMQ output. Model-layer logits and continuation parity are
still required before production integration.

## Results

The values below are the B512 rows from the exact production row-128 bake-off.
Times are median microseconds from the benchmark's seven timed samples.

| tensor | shape (rows × columns) | type | resident MMQ | direct FP16 total | MMQ / FP16 |
|---|---:|---|---:|---:|---:|
| `blk.0.ffn_gate.weight` | 17408 × 5120 | Q4_K | 13212.3 | 10207.7 | 1.29× |
| `blk.0.ffn_up.weight` | 17408 × 5120 | Q4_K | 13208.8 | 10296.3 | 1.28× |
| `blk.0.ffn_down.weight` | 5120 × 17408 | Q4_K | 13920.5 | 10415.0 | 1.34× |
| `blk.0.ffn_down.weight` | 5120 × 17408 | Q6_K | 13686.7 | 10493.4 | 1.30× |
| `blk.0.ssm_out.weight` | 5120 × 6144 | Q5_K | 4819.4 | 3464.3 | 1.39× |

Direct resident expansion errors were zero at the sampled rows for these model
tensors. The Q5 case used `resident_repack_us=473.3 us`; its B512 GEMM was
2991.0 us. Q4/Q6 expansion was approximately 1.27–1.34 ms for the large
FFN shapes.

## Interpretation

The earlier ~2.9× Q4 Down result came from a non-production MMQ mapping. With
the qualified row-128 mapping, the real architecture gap is approximately
1.28–1.39× at B512. That is still a consistent win, but it is not sufficient
to claim that FFN migration alone reaches 180 or 200 tok/s. The Q5 SSM output
case and both Gate/Up orientations show that the candidate is viable across
the tested Q4/Q5/Q6 mix, while the modest ratios make end-to-end coverage of
attention, output projection, and normalization essential.

## Decision

**KEEP as the prefill projection candidate; stop MMQ micro-tuning for now.**
Do not replace the production path yet: integrate one recurrent layer and one
attention layer, validate against the canonical/reference computation, and
measure whole-model P512 with continuous clock telemetry.

## Follow-up

1. Add a one-layer resident-MMQ-to-FP16 execution path with reusable scratch.
2. Compare layer tensors, logits, top-k ordering, and short/long continuation.
3. Extend the path to all eligible P512 projections and remeasure latency and
   VRAM before any Gate/Up fusion work.
