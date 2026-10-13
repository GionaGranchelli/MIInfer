# EXP-0405 — M31 query-tiled Q8 attention with K-fragment reuse

**Status:** FOCUSED KERNEL PASSED — FULL-MODEL GATE PENDING

**Baseline:** `2779a3c` on branch `perf/m31-perf-recovery-0001`

## Objective

Reduce large-query causal GQA attention time while retaining the production
Q8/Q8 KV representation. The 64K late-tile profile in EXP-0404 attributes
58.448% of the 5.518 s FP16-KV tile to causal attention. The existing Q8
Split-K control at `P=65024, S=512` took 1,587.699 ms for 32 splits; 64 splits
was only 6.3% faster with twice the scratch, so it was not promoted.

## Hypothesis

The existing KQ-fragment-reuse schedule handles 16 adjacent query tokens and
two query heads per KV group. Applying it to Q8 KV should reuse each staged K
fragment across those queries and dequantize K/V while staging into LDS,
avoiding the control's independent per-query cache traversals without adding a
full FP16 copy of the KV cache.

## Independent variable

Only the attention execution schedule changes: existing Q8/Q8 Split-K versus
the same 16-query K-fragment-reuse mapping with on-the-fly Q8-to-FP16 LDS
staging. Keep `P=65024`, `S=512`, 32 splits, Q/Gate/cache operands, and the
model/runtime configuration fixed. Keep the outer prefill macro tile at 512.

## Gates

- **Resource:** compile the `gfx906` specialization and record VGPR, SGPR,
  LDS, scratch, and spills. Unexpected spilling rejects the candidate before
  timing.
- **Correctness:** compare candidate against the existing Q8 Split-K result at
  `P=65024, S=512`; require `max_abs <= 0.01`, cosine `>= 0.9999`, and finite
  outputs.
- **Performance:** five interleaved timings on the same MI50; keep only if the
  candidate median is at least 10% faster. Reject at 3% slower or worse.
- **Product:** if focused gates pass, enable only for supported Q8/Q8 full
  prefill tiles and run one matched shorter full-model continuation before
  attempting the expensive 64K product workload.

## Kill / recovery

Stop candidate timing on invalid output, a HIP error, unexpected spills, or
unsafe GPU temperature. Keep the production dispatch unchanged on any failed
gate. If dequantization cost erases the reuse gain, change the dataflow rather
than adding split-count or macro-tile variants to this candidate.

## Focused implementation and result

Commit `93a06a0` adds the Q8 specialization of the existing K-fragment-reuse
kernel and extends `bench/m31_q8_fp16_tiled.cpp` to compare it with the direct
Q8 control. The K/V rows are dequantized into the existing LDS staging tile;
there is no full-size FP16 cache copy and no production dispatch change.

The pinned ROCm 7.2.1 container (`sha256:447907c697fb2c39ce9657401e354a70753d5672ad955d4ceb12de976b91224a`)
compiled the candidate for gfx906. Compiler resource remarks for the Q8/Q8
specialization reported 72 VGPR, 49 SGPR, 0 bytes/lane scratch, zero SGPR and
VGPR spills, and 3 waves/SIMD. The launch requests 27,008 bytes of dynamic LDS
per block; static LDS reported by the compiler was zero.

On Machinist GPU2 (`0000:88:00.0`), with identical synthetic Q/K/V operands at
`P=65024, S=512` and 32 splits, candidate output matched direct Q8 within
`max_abs=0.0000573`, cosine `0.9999782`; outputs were finite. Five interleaved
GPU-event samples were:

- Direct Q8 Split-K: `1650.146, 1637.318, 1633.277, 1612.331, 1601.343 ms`; median
  `1633.277 ms`.
- Query-tiled Q8 with fragment reuse: `222.824, 196.849, 196.805, 196.715,
  197.111 ms`; median `196.849 ms` (8.297x faster).

The thermal supervisor sampled seven times, peaked at 54 C, and reported
`supervisor_failed=0 benchmark_exit=0`. The focused kernel gate passes. This
does not prove full-model logits, token behavior, or end-to-end prefill; keep
the route opt-in until a matched full-model Q8/Q8 check passes.
