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
The updated gfx906 target compiles with the pinned ROCm 7.2.1 image. Its Q8
specialization uses 88 VGPR, 49 SGPR, no scratch or spills, two waves/SIMD, and
requests 29,056 bytes of dynamic LDS. Neither this precision change nor the
3-split setting has GPU correctness or performance evidence yet. Keep the
whole route opt-in.

The failed HIP copy was confirmed to target the intended card: inside the
container, `ROCR_VISIBLE_DEVICES=2` exposes GPU UUID
`GPU-2152414173497dd3` (GPU2, BDF `0000:88:00.0`, gfx906). `rocminfo` can
enumerate the agent, but both the model run and a 4 KiB vector-transfer run
fail when executing HIP work. Once GPU2's HIP copy path works again, resume
with the 1,024-token matched correctness check, then the same 8K gate only if
the short check passes.

## ROCm 7.14 recovery and larger attention-batch check

The ROCm 7.14 gfx906 image restored HIP execution on isolated GPU2. With the
same synthetic 64K endpoint (`capacity=65536`, `prefix=capacity-queries`), the
focused harness now accepts query batches of 512 and 1024. Five alternating
direct-Q8 / exact-tiled-Q8 timings were collected per batch under the 80/85 C
thermal guard. The harness confirmed BDF `0000:88:00.0`; both candidates
returned finite values, and exact tiled Q8 matched direct Q8 (`max_abs=0`,
cosine `1.0`).

| Queries | Direct Q8 median (range) | Exact tiled Q8 median (range) | Direct / tiled |
| ---: | ---: | ---: | ---: |
| 512 | 480.850 ms (474.443–494.086) | 671.403 ms (639.030–688.409) | 0.716x |
| 1024 | 967.073 ms (955.322–991.760) | 1278.403 ms (1267.027–1293.650) | 0.756x |

Direct attention cost per query was 0.939 ms at 512 and 0.944 ms at 1024,
effectively unchanged. The exact tiled path improved from 1.311 to 1.248 ms per
query, but remained 32% slower than direct Q8 at 1024. Junction temperature
peaked at 55 C and 59 C respectively; no guard fired. The image/toolchain is
ROCm 7.14 / Clang 23, so these measurements are valid as a same-stack A/B and
must not be merged numerically with ROCm 7.2.1 records.

This only tests attention query-batch scaling; it does not exercise the
full-model macro scheduler. It provides no evidence that a 1024-token
full-model tile reduces dispatch overhead. No utilization or instruction
profile was collected, so the bottleneck within the attention kernels remains
unattributed. Keep exact tiled Q8 rejected for promotion. Next, fix the
full-model Q8 dispatch/workspace contract before attempting a 1024-token
physical tile: the default attention-layer route selects Q8 Split-K only for
`token_count <= 512` and falls through to FP16 online attention above that;
the opt-in Q8 tiled route is separate and currently slower. Several other
primitive/workspace limits are also hard-coded at 512.

## Exact Q8 QK/V candidate

Source comparison showed that the direct Q8 kernel uses FP32 queries and
multiplies the quantized K values by their stored row scales during QK, while
the first tiled Q8 candidate rounded both Q and dequantized K to FP16. The
follow-up specialization keeps Q and K in FP32/int8 form for QK, stages the
transposed int8 K tile and its row scales in LDS, and accumulates V directly
from int8 values with row scales into FP32 outputs. It still reuses each K/V
tile across neighboring queries; the FP16 specialization is unchanged.

Commit `f394f87` removes the now-unused half-dequantization helper. The
correctness-gated benchmark target builds with the pinned ROCm 7.2.1 image.
Compiler remarks for the Q8 specialization report 116 VGPR, 60 SGPR, no
scratch or spills, and two waves/SIMD. The launch requests 45,568 bytes of
dynamic LDS per block. This higher resource use is within MI50 LDS capacity,
but the candidate has no GPU correctness or timing result because GPU2's HIP
execution fault remains unresolved. It remains opt-in.
