# M31-0004 — GDN checkpoint lifecycle

## Ownership and purpose

`ReusableContext` owns one reusable checkpoint: 48 GDN states and 48 convolution
histories. `generate` captures it after completed 512-token boundaries when
prefix caching is enabled. Each boundary overwrites the prior slot; this is
not a collection of historical checkpoints. On successful completion, the
full prompt is captured too if it ends off-boundary, while an already-captured
final boundary is not copied twice. A later exact-prefix hit restores the
saved GDN state before suffix execution. A snapshot is separate immutable
backing and must not alias mutable active recurrent or KV state.

The boundary captures are not dead copies: retaining the latest completed
boundary preserves a resumable prefix if a later chunk is cancelled or fails.
Because storage has one slot, it preserves only the latest captured prefix,
not all intermediate lengths. Deleting these captures or replacing them with
mutable aliases would lose that recovery point. Cold execution without prefix
caching allocates no GDN checkpoint buffer; allocation is lazy.

## Copy ledger

Each capture performs 96 device-to-device copies: state and convolution history
for each of 48 recurrent layers. The total is 158,859,264 bytes (151.5 MiB) per
boundary. When final hidden is available, one additional 20,480-byte copy is
made. Captures are queued on the supplied stream; the capture method itself
does not synchronize per boundary.

| Prefix length | Boundaries | GDN D2D calls | GDN bytes |
| ---: | ---: | ---: | ---: |
| 8K | 16 | 1,536 | 2,541,748,224 |
| 32K | 64 | 6,144 | 10,166,992,896 |
| 64K | 128 | 12,288 | 20,333,985,792 |
| 128K | 256 | 24,576 | 40,667,971,584 |

This cumulative traffic is the sum of repeated boundary snapshots, not peak
resident memory. Snapshot capture and disk-session persistence have distinct
copy paths and are not included in this table.

## Change made

No GPU checkpoint copies were removed: the available evidence does not prove
that any intermediate boundary is unused. The implementation now has an
explicit `save_extension` operation for a caller that already knows the exact
cached boundary and supplies only the newly appended tokens. It still performs
the same 96 GDN copies and optional hidden copy. It reduces host token copying
and hashing, not checkpoint traffic.

The shared runtime invalidates the resident marker before state mutation and
only restores it after a successful checkpoint capture/restore. Successful
prefill also saves the complete prompt when it ends off-boundary, including
short prompts; an already-captured final boundary is not copied twice. This
fixes both missing short-prefix caches and the 513-token stale-state case.
After saving token 512, the final one-token forward cannot leave the 512
checkpoint falsely marked as live; successful completion captures token 513,
while interruption before that save leaves the 512 checkpoint nonresident and
restorable.

## Semantics and risks

- The ordinary `save` path remains available for arbitrary prefixes and
  recomputes its full fingerprint.
- `save_extension` checks the expected prior boundary and requires a non-empty
  appended range. Its caller must prove that range is the next portion of the
  same prompt; production use is limited to the monotonic generation loop.
- Cancellation after a forward but before the callback/checkpoint leaves the
  older checkpoint retained but nonresident, so it can be restored safely.
- Snapshot backing remains immutable; this change does not introduce aliasing
  or copy-on-write changes.
- Conservative invalidation from mutable model accessors can cause an extra
  restore after a read-only use of those APIs. Correctness is preferred over
  assuming a mutable reference was not changed.

No device-copy reduction or hardware correctness claim is made.
