# EXP-0398 — M31 Q8 cold-prefill attribution

**Status:** IN PROGRESS  
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

Pending the focused MI50 run.
