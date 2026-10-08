# M31-0003 — Attention and KV investigation

## Scope and evidence

This is source analysis plus existing artifacts only; no GPU work was run.
The current Qwen 27B topology has 16 GQA layers, each with 24 query heads,
4 KV heads and head dimension 256. KV is FP16 in the production/M31 path.
Each query head uses the K/V pointer for its mapped KV head. Thus six query
heads logically consume the same KV head data, but execute independent
per-query-head kernel work. This proves repeated loads/address work in the
kernel structure, not sixfold HBM traffic: device cache behavior is unmeasured.

The attention kernels read `DeviceKvPool` storage through the layer's KV view;
they do not first materialize a contiguous copy for attention. Each attention
layer owns a separate backing pool. The full causal prefill attention work is
quadratic in prompt length; Split-K partitions the reduction and scratch, not
the logical Q×K work.

## Long-context traffic model

For one decode token, scanning every prior K and V element once per shared KV
head gives the following logical read estimate across all 16 layers:

| Context | Once per KV head | Once per query head |
|---:|---:|---:|
| 16K | ~1 GiB | ~6 GiB |
| 128K | ~8 GiB | ~48 GiB |

These are source-derived bytes (`layers × KV heads/query heads × context ×
head dimension × K/V × FP16`), not measured DRAM traffic. Cache reuse, transaction
efficiency and achieved bandwidth are unknown. This is why GQA sharing is a
credible target, but not a justified kernel rewrite by itself.

## Dispatch and historical experiments

Iteration 26 (`V2-0043`) is opt-in in the low-level helper, but the first-party
`run`, `chat` and `serve` CLI setup forces `MIINFER_V2_0043_GQA_ATTENTION=1`.
It can be selected for eligible cold-prefill chunks after position zero: FP16
KV, no device prefill-state pointer, and a token count that is a positive
multiple of 16. The first cold chunk and graph-backed persistent suffixes do
not use it. M31 checkpointed benchmark source does not set this variable and
its results do not record the inherited environment, so its exact selection is
unverified.

The M31 Split-K microbenchmark calls a different kernel from V2-0043. Its
three-way/32-way scratch is about 38/406 MiB, respectively. The historical
32-split result was about 1.43× faster for a synthetic 16-token query at 32K
and 64K, not a production 512-token tile or 128K result. EXP-0360's six-wave
LDS GQA-sharing attempt was correct but slower (about 0.51–0.67× control); do
not repeat it without a materially different design or evidence.

## Decision

**Confirmed:** context-dependent KV scan grows linearly per decode token;
prefill attention grows quadratically with prompt length. **Strong hypothesis:**
sharing K/V work across the six mapped query heads could reduce redundant work.
**Not established:** actual HBM amplification, the 32K/64K full-model bottleneck,
or a faster gfx906 kernel. Keep current kernel selection; first obtain short,
matched component evidence in a later GPU-authorized task.

Cheapest validation is an offline address/traffic model and existing kernel
correctness tests. Performance, cache behavior and the best grouping strategy
remain GPU-dependent.
