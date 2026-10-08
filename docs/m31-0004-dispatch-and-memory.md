# M31-0004 — Dispatch and memory findings

## Per-tile execution model

The cold production path divides prefill into tiles of at most 512 tokens. For
each tile, it queues token H2D, embedding, 16 topology blocks, final RMS norm,
and host bookkeeping. Each topology block includes three recurrent layers and
one attention layer. Each recurrent layer issues about 20 HIP launches for Q4
or 21 for the Q6K QKV variant, giving 960–1,008 GDN-related launches per full
tile. These counts are source-level launch sites, not a GPU trace.

Incoming and outgoing recurrent state pointers alias in production, so the
recurrent layer skips its optional state-preservation D2D copy. Workspace is
reused; no per-tile HIP allocation/free cycle was found. The GDN kernel reads
its register-sharded state once before the token scan and writes it once after.

## Synchronization and transfers

Ordinary `forward` queues work on its stream without a per-layer or per-tile
sync. The generation path synchronizes for logits/token sampling and at explicit
reuse boundaries. The profiled path introduces per-layer event/synchronization
overhead and must not be treated as ordinary throughput. Prefix restore uses
96 asynchronous GDN copies followed by a synchronization before suffix work
where required. This milestone does not remove that barrier: stream ownership
and timing behavior differ across callers and persisted-session paths, and no
measurement demonstrates it is redundant.

Checkpoint capture queues 96 D2D calls and optionally one final-hidden copy per
512-token boundary. At 128K, that is 24,576 GDN D2D calls and 40,667,971,584
aggregate GDN bytes. These copies preserve intermediate-prefix state; the
milestone leaves them unchanged.

## Host bookkeeping optimization

Repeated full-prefix token copies and FNV-1a scans were avoidable when the
production loop has already established a monotonic prefix extension. The
first cold boundary uses ordinary `save`; later boundaries use `save_extension`
with only the newly appended 512 tokens. Reused suffix execution extends only
when the cached checkpoint length equals the preceding boundary; otherwise it
falls back to ordinary `save`.

At 128K, hashing visits 16,842,752 tokens across the 256 boundaries before and
131,072 after. Full-prefix vector assignment is likewise replaced by appending
the 131,072 new token payload; vector capacity growth may relocate prior
elements occasionally. The hash work falls by 16,711,680 visits (about 99.2%),
and explicit token payload copying becomes linear rather than re-copying each
full prefix. The CPU saving has not been timed.

## Changes not made

- No HIP synchronization was removed.
- No GDN kernel launch was removed or fused.
- No state checkpoint D2D call or byte was removed.
- No new scratch buffer, allocation policy, or graph-capture boundary was
  introduced.

Next best GPU optimization candidate: test cooperative Q/K reuse across GDN
state-column tiles with a focused kernel benchmark and state/output parity
checks. The existing TC4 result argues against switching layouts without new
evidence.
