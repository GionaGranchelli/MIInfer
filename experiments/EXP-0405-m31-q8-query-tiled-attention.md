# EXP-0405 — M31 query-tiled Q8 attention with K-fragment reuse

**Status:** IMPLEMENTATION IN PROGRESS

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
