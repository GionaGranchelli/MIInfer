# M31-0005 — Dispatch and synchronization audit

Status: source audit complete; GPU timings pending.

## Per-token graph sequence

The normal decode loop performs, for every generated step:

1. HIP Graph launch for embedding, all 16 topology blocks, final norm, and
   logits.
2. Device-to-host copy of the full logits vector.
3. `hipStreamSynchronize` so CPU sampling can consume those logits.
4. CPU sampling with current repetition/penalty/top-k/top-p/stop semantics.
5. A small host-to-device `DeviceDecodeState` upload when another graph
   iteration will run.

No per-layer host synchronization appears in the ordinary graph path. Attention
stage 1 and stage 2 are ordered on their stream. The full-logits copy and
synchronization are not removable without changing where sampling occurs or
preserving the same sampling behavior elsewhere.

The final loop iteration currently uploads the next 64-byte decode state even
though no later graph iteration consumes it. The next `generate` call
initializes decode state before launching its graph, so this is unnecessary
traffic within that call. It is one 64-byte transfer per generation, negligible
against context-scaled KV traversal; it is not a material latency fix and has
not been changed.

## Attribution limits

There is no source or artifact evidence of a per-layer sync in unprofiled graph
decode. The profiled route deliberately adds event/synchronization work and
must not be used as the production latency model. Graph capture happens once
per model instance, outside the per-token loop. Host sampling and full-logits
transfer are fixed per token; their cost does not intrinsically scale with KV
length, though total decode latency does.

The exact component cost of graph launch, logits transfer, synchronization,
sampling, and state upload is unknown. Do not assign the 8K→128K latency delta
to dispatch based on source inspection alone.

## Follow-up

If GPU stop safety is independently established, add narrow existing timer or
event measurements around one short graph decode and separate CPU sampling
wall time from device elapsed time. Preserve sampling semantics and graph
replay; do not start with a broad profiler campaign.
