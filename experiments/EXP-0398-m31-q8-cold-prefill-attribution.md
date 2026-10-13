# EXP-0398 — M31 Q8 cold-prefill attribution

**Status:** LEARN

**Baseline:** `88fd63eb0cf8b765f37e5f52168c80783df62e7b`
**Question:** Does the current Q8/Q8 causal-attention stage account for a material portion of late-context PrefillV2 tile time?

## Evidence motivating the measurement

The M31 product record reports a 78,245-token cold request at 26.91 tok/s,
while a prior GPU2 run processed 73,099 tokens at 31.81 tok/s. Neither report
contains a stage split. Historical P8192 measurements attributed the decline
from 223.1 tok/s at P512 to increasing causal-attention work, but those results
do not isolate the current Q8/Q8 path at a late-context tile.

At the selected source baseline, Q8/Q8 attention dispatches one CTA per query
position, query-head pair, and KV split. Each CTA traverses its assigned part
of that query's causal prefix. Neighboring queries therefore issue independent
loads for overlapping KV ranges. The reported product model uses Q8 KV, so
this is the path to measure first. Earlier rejected FP16 query-tile candidates
are not treated as evidence for or against this Q8-specific execution shape.

## Hypothesis

When profiling the final 512-token tile of an 8192-token cold prefill, Q8/Q8
causal attention is a material part of GPU time and is worth a dedicated
batched attention implementation.

## Measurement contract

- **Objective:** Attribute the late-context full-model tile before changing its kernels.
- **Current bottleneck evidence:** Historical cold-prefill scaling and the current per-query Q8 Split-K dispatch are suggestive but do not quantify this path.
- **Reference:** Existing `PrefillV2AttentionLayer::forward_profiled` event breakdown, selected under the same Q8/Q8 cache mode as the product path.
- **Difference under test:** Only event reporting in the existing profiled call path; normal `forward()` execution is unchanged.
- **Expected maximum end-to-end impact:** The measured Q8 attention share of the profiled tile; no speedup is claimed by this instrumentation.
- **Independent variable:** Q8/Q8 attention stage duration at context position 7680.
- **Resource gate:** No new GPU kernel, allocation, or production hot-path work; use one idle MI50 with the model and Q8 cache.
- **Correctness gate:** Process the preceding 7680 tokens through ordinary forward; profile the final 512; require finite final hidden values.
- **Performance gate:** Compare Q8 attention event time against full profiled tile time and report the layer breakdown. Do not treat event-instrumented wall time as production latency.
- **Kill criteria:** If the profiled path differs from the production Q8 dispatch or state/output is non-finite, discard this attribution and fix the profiler contract first.
- **Success criteria:** A reproducible stage split establishes whether Q8 attention has enough Amdahl ceiling to justify a candidate; if not, redirect to the measured larger stage.
- **If rejected:** Do not build the Q8 attention candidate; select the next bottleneck from the reported breakdown.

## Implementation

The profiled topology block now exposes the existing attention-layer phase
timings, and `miinfer-m31-prefill-attribution` fills the context with ordinary
512-token forwards before profiling only the last tile. It defaults to 8192
tokens and Q8/Q8 KV. The output checks the final hidden vector for finite values.

## Results

The profiler was built with the pinned ROCm 7.2.1 image and run on Machinist
GPU2. It processed 7,680 prefix tokens using ordinary forward calls, then
profiled the final 512-token tile:

| Item | Result |
| --- | ---: |
| Source | `4d169b4` |
| Model SHA-256 | `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169` |
| GPU BDF / unique ID | `0000:88:00.0` / `0x2152414173497dd3` |
| Profile tile base / shape | `7680` / `512` |
| Full profiled tile GPU time | `4944.550 ms` |
| All 16 GQA block time | `3160.662 ms` |
| Q8/Q8 causal-attention time | `2644.759 ms` |
| Attention share of full tile | `53.488%` |
| Attention share of GQA blocks | `83.68%` |
| Final hidden values | finite |
| Peak sampled junction temperature | `61 C` (49 samples; initial `29 C`) |

This is direct stage timing on the current Q8/Q8 PrefillV2 path. Attention has
a material Amdahl ceiling at this shape, so the next attempt is a Q8-specific
two-query schedule: map one wave to each of two neighboring query positions,
stage each shared 32-position Q8 K/V tile once in LDS, and keep one online
softmax state per wave. The earlier FP16 multi-query candidates do not use this
data type or state mapping and do not decide this candidate.

The measurement covers one late-context tile, not a full 64K cold request. It
selects the next implementation; it is not a performance win or product-level
speedup claim.
