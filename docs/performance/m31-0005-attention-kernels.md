# M31-0005 — Attention kernel audit

Status: source audit complete; candidate kernel changes remain unvalidated.

## Graph decode kernel

The normal M31 benchmark enables HIP Graph. Its GQA path calls
`launch_qwen35_tiled_online_attention_quant_dynamic` from
`src/prefill_v2/attention_layer.cpp`. The current FP16 implementation launches
stage 1 as a fixed grid of 24 query heads by 64 splits, one Wave64 CTA per
head/split, then stage 2 reduces split maxima, sums, and accumulators. Each
stage-1 CTA independently loads its query vector and walks its assigned KV
range, using online softmax. Query head `h` maps to KV head
`h / (24 / 4)`; six query heads therefore map to each KV head. No cross-query
head reuse is implemented in this kernel.

`MIINFER_ATTENTION_SPLITS` can set a fixed active split count. Without the
override, device code selects 4/8/16/32/64 active splits across increasing
length ranges. The graph grid remains 64; inactive CTAs return. Thus 8K uses
16 active splits (48 early returns/head), while 128K uses all 64. The stage-1
and stage-2 launches are ordered on the same stream without an intervening host
sync. Device-global scratch is approximately 64 * 24 * 256 * 4 = 1.5 MiB for
accumulators plus 64 * 24 * 2 * 4 = 12 KiB for max/sum metadata.

## Separate eager/suffix path

When `decode_state == nullptr`, the attention layer calls the quantized
suffix Split-K kernel using workspace allocated from the prefill workspace
manager. That is not the normal M31 frontier path. The workspace reservation
formula is `splits * 512 * 24 * (256 + 2) * 4` bytes: 32 splits reserve
405,798,912 bytes (387 MiB); 3 splits reserve 38,043,648 bytes (36.3 MiB).
The >66,000-capacity rule selecting 3 splits reduces eager workspace; it does
not change graph decode's dynamic 64-grid policy.

The eager caller previously passed `position + 1`; the kernel computes
`base_position + token + 1`, so with a one-token suffix this included one
future KV slot. The caller now passes `position`. The direct adversarial HIP
test uses a future-slot poison value to assert it cannot affect attention.
This fixes the eager bound only; the graph path already takes its inclusive
length from `DeviceDecodeState::position + 1`. The GPU regression test is not
yet executable in this environment.

## Unproven opportunities

The best first tuning candidate is to benchmark active split counts on the
actual graph path. No split count is promoted: changing split scheduling
changes reduction order and occupancy and needs parity plus timing on gfx906.
Grouping multiple query heads per CTA could reuse KV data, but raises register,
occupancy, and reduction risks. Historical cross-head experiments were neutral
or regressed on different geometry; they do not support a new production
kernel. Stream-K or a different token tile may help, but no local measurement
establishes that.

No production graph kernel was replaced and no speedup is claimed.
