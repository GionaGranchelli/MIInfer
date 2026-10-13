# EXP-0399 — M31 Q8/Q8 two-query shared attention

**Status:** IN PROGRESS

**Baseline:** `88fd63eb0cf8b765f37e5f52168c80783df62e7b`
**Current candidate:** Q8/Q8 Split-K attention, two adjacent query positions per CTA.

## Objective

Reduce the 2.645 s Q8/Q8 causal-attention stage measured in the final P8192
tile, while preserving the existing Q8 dense-GQA result and the 32-way split
reduction contract.

## Hypothesis

The current grid independently launches one query position at a time. Mapping
each 32-lane half-wave to a different adjacent query position for the same Q
head allows both waves to consume a single 32-position Q8 K/V tile staged in
LDS. Each half-wave retains only one query's online-softmax state, avoiding the
per-thread state multiplication that caused the earlier register-resident
cross-query experiment to spill.

## Evidence and reference

- EXP-0398 measured Q8/Q8 attention at `2644.759 ms` (`53.488%` of the full
  late-context tile) on one MI50.
- The production reference is `launch_qwen35_splitk_suffix_attention_quant`
  with Q8 K, Q8 V, and 32 KV splits.
- Earlier FP16 query-tile candidates are not comparable: they used another
  cache representation and assigned several query accumulators to each wave.
  This candidate assigns one query to each half-wave and uses 16.25 KiB LDS to
  share Q8 K/V bytes.

## One variable

Only the execution mapping changes: independent single-position CTAs become
two-position CTAs that reuse staged Q8 K/V tiles. Head dimension, Q8 formats,
online softmax, causal mask, 32 Split-K partitions, and stage-two reduction stay
the same.

## Gates

- **Compiler/resource:** gfx906 code object, 64 threads, expected `16,640 B`
  LDS, zero VGPR/SGPR spills, and no unexpected private memory.
- **Correctness:** On fixed Q8/Q8 operands at `P=8192, S=512`, compare output
  against the current split-K path; require `max_abs <= 0.01` and cosine
  similarity `>= 0.9999`. The existing small GPU attention test remains the
  independent Q8-vs-FP16 quality check.
- **Performance:** Five interleaved baseline/candidate event samples at the
  same `P=8192, S=512` shape. Require at least 10% median improvement before
  adding a runtime selector or testing a full-model tile.
- **Kill:** Reject on incorrect output, any unexpected spill/private state,
  or more than 3% slowdown. Do not widen the query tile to rescue a failed Q2
  result without new machine-level evidence.
- **Success:** Correctness and resource gates pass, then candidate median is at
  least 10% faster. Only then route the model's Q8/Q8 path through an opt-in
  selector and measure one full-model late-context tile.
- **If rejected:** Keep the production path unchanged and change the algorithm
  family, using measured Q8 traffic and compiled ISA as input.

## Implementation

An isolated kernel and a single-shape interleaved comparison executable have
been added. The production attention dispatcher still selects the existing
kernel; no runtime behavior changes before the gates pass.

## Results

Pending compile-resource, numerical, and MI50 timing gates.
