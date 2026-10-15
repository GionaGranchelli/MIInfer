# EXP-0468 — Sparse local-window plus absolute anchors (GPU2 kill screen)

## Claim and scope

Test one explicitly approximate GQA6 Stage1 candidate. For causal key length
`T <= 2,048`, attend to every key. For `T > 2,048`, attend to the latest
2,048 keys plus absolute positions `0, 512, 1,024, ...` strictly below
`T - 2,048`. This retains all canonical Q8 KV entries and persistent-state
positions but changes which keys contribute to attention. It is experimental,
opt-in, and cannot replace the exact production default based on this screen.

## Frozen comparison

- Source base: `c7806fcca074f04fc51eac1ba2263d534e318be4`.
- Control and candidate: same pinned ROCm 7.2.1 binary, model, restored
  73,099-token session, tokenizer-backed 1,024-token suffix, GPU2 BDF
  `0000:88:00.0`, 32 Split-K partitions, existing Stage2 and one HIP Graph
  replay tile. GPU0 remains excluded.
- Control: existing exact direct-global Q8 GQA6 Stage1.
- Candidate: opt-in sparse selector, same Q8 cache/scales, FP32 per-key
  recurrence, partial workspace, and Stage2.

## Implementation gates (before GPU)

1. Selector maps a sorted, unique set; for `T <= 2,048`, it is exactly
   `[0, T)`. For larger T, every key in `[T-2,048, T)` appears once and each
   512-position anchor strictly before that window appears once.
2. A host assert oracle covers T at 1, 2,047, 2,048, 2,049, 73,099,
   anchor/window boundaries, ordinal ordering, and disjoint 32-way ceil-sized
   split partition including empty splits.
3. Candidate graph identity is distinct from direct-global. Route tracing must
   prove the requested candidate and `graph_state=1`; otherwise stop.
4. Pinned target builds for gfx906. Inspect VGPR, SGPR, spills, scratch, LDS,
   workgroup, and disassembly for absolute Q8 KV and scale indexing before
   using GPU2. GPU2 preflight verifies BDF and temperatures; warn at 80 C and
   stop at 85 C.

## GPU kill-screen gates

One same-binary control/candidate pair only. Both must restore 73,099 and end
at 74,123, process all 1,024 suffix tokens without truncation, replay the
captured graph, and show the expected route in all 16 GQA layers. Record finite
logits and recurrent state, max-absolute and cosine deltas, generated tokens,
per-route suffix wall time, and GPU telemetry. Candidate passes this screen
only if no output is non-finite, generated behavior is not catastrophic, and
suffix time is at least 10% lower than control. Catastrophic means failure to
produce a completion, invalid state/logits, or immediate nonsensical/repetitive
behavior; token equality alone is not a correctness proof for approximate
attention.

If it fails speed or outputs catastrophically, stop. If it passes, report that
a real coding-agent task with facts older than the local window remains
required before promotion. No parameter sweep and no cold 64K run is in scope
for this screen.

## Decision

PENDING implementation, compiler/resource/ISA gates, and the one GPU2 screen.
