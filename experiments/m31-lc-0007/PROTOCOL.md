# M31-LC-0007 — Q4_K Affine-Sum Contract Alignment

## Scope

Test whether changing only the MMQ Gate/Up Q4_K minimum-correction sum restores fused/MMQ exact-token parity on the frozen 2K graph workload. Keep fused as the default, leave the normal MMQ sum path unchanged, preserve quantized integers, scales and weight layouts, and do not begin the context ladder.

## Frozen identity

- Base: `d507c4f`; executable source: `8a9caec3366512e459a93bad68adde181225304b`.
- Z840 MI50 `0000:06:00.0`, image `sha256:bdc5ed42c985a6a333083e023225f248825a6e1511f38b6b50fbc7d5ace3fe9e`.
- Model SHA-256: `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.
- 2,048 prompt tokens, 2,304 capacity, five greedy output tokens, same reset state and graph mode.
- The first decode replay consumes token `220`; compare this common prefix and do not force a divergent later history.

## Sequence

1. Reproduce graph-mode control and MMQ raw logits on the base binary; compare each hash with LC-0006.
2. Capture layer-0 graph-safe shared input/state, post-normalized input, packed Q8 blocks, MMQ Gate/Up outputs, activation, and layer output for both routes.
3. Independently verify packed integers/scales and compute both correction sums and Q4_K projection references from the captured inputs and canonical weights.
4. Add an opt-in diagnostic selector used only by MMQ Gate/Up. Compare existing original-input sum with the requested half-rounded `d * sum(q)`; do not alter default behavior or other MMQ projections.
5. On the same graph workload compare fused, original MMQ, and aligned MMQ: both Gate/Up outputs, activation, layer output, all five raw-logit vectors, token IDs, and allocation/VRAM records.
6. If parity holds, validate both arithmetic contracts against independent projection references and the CPU full-model reference. If it fails, trace the next earliest differing operation and stop.
7. Build all targets; run full configured CTest, checksum and secret scans, thermal guard, and Git hygiene checks.

## Acceptance

Candidate passes only if aligned MMQ produces the fused token sequence on this workload, improves or preserves independent numerical fidelity, retains the MMQ VRAM advantage, and leaves default fused behavior unchanged. A 2K pass is not long-context qualification.
