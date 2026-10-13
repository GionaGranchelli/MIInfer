# EXP-0405 — M31 query-tiled Q8 attention with K-fragment reuse

**Status:** FOCUSED KERNEL PASSED — INITIAL FULL-MODEL CORRECTNESS FAILED — RETEST BLOCKED BY MACHINIST GPU2 HIP FAULT

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

## Full-model integration failure and precision follow-up

On a tokenizer-backed 8,192-token prompt at capacity 65,536, three interleaved
Q8/Q8 pairs on GPU2 (`0000:88:00.0`) produced a repeatable logit mismatch:
`max_abs=0.657086372`, cosine `0.999395058`; top-1 and second token IDs matched.
Median prefill was 49.996 s for the control and 40.537 s for the tiled route
(1.233x), but the candidate fails the predeclared `cosine >= 0.9999` gate and
must not be promoted.

The Q8 tiled route had used 32 splits while the existing full-model FP16 tiled
route uses 3. Commit `0aabba9` changed only the opt-in Q8 route to 3 splits to
test whether partial-output grouping caused the accumulated difference. That
candidate compiled. Its tokenizer-backed 1,024-token full-model rerun did not
reach model output: ROCr reported `Memory critical error ... Reason: Memory in
use` on GPU2. A separate 1,024-element `miinfer-bench` run, which transfers
4 KiB vectors, failed with the same ROCr fault and exit 139. GPU2 telemetry
remained available at 30–31 C. The live MIInfer service on GPU0 was using
29,304,508,416 bytes; it was left untouched. This is a current Machinist
GPU2 HIP execution blocker, not evidence about the 3-split candidate.

Commit `5646459` adds a second opt-in precision change: Q8 attention weights
and V accumulation remain FP32 instead of being rounded into half fragments and
half accumulators. The FP16 specialization retains its prior half behavior.
The updated gfx906 target compiles with the pinned ROCm 7.2.1 image. Neither
this precision change nor the 3-split setting has GPU correctness or performance
evidence yet. Keep the whole route opt-in. Once GPU2's HIP copy path works
again, resume with the 1,024-token matched correctness check, then the same 8K
gate only if the short check passes.
