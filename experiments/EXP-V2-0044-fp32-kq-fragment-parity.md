# V2-0044 — Precision-preserving KQ fragment on the accepted tile path

## State

PRIMARY candidate 1; parity gate only until the complete real-model token
trajectory passes. The immutable control is canonical merge
`e0649c589790f0267a8f1f45b5332532a95ae13a` / production source checkpoint
`e68c0f20`. This experiment does not change the default path.

## Objective and measured motivation

V2-0043 closed PASS on iteration 26. The accepted path remains end-to-end
faster than the pinned references at P8192, while its measured main-attention
device time is still about 1.91× pinned mx. Do not reopen its rejected
geometry, unroll-only, padding, or split-order variants without a materially
new formulation.

The exact real-input attribution in
[`EXP-V2-0043`](EXP-V2-0043-reference-shaped-gqa-attention-rewrite.md)
isolated the first tolerance-breaking value at layer 3, chunk base 512, local
query token 1 / absolute position 513, Q head 20, dimension 45. QK scores
matched exactly; online max matched at `8.30334`, sum error was `2.38419e-7`,
and reconstructed FP32 normalized weights differed by at most `7.45058e-9`.
The reference-shaped candidate then rounded tile-local weights to FP16 before
V. That fragment introduced max normalized-weight error `2.46093e-5` and
gated V/output error `2.92063e-5`, just beyond the established
`2.7563e-5` envelope. This is new causal evidence for a precision-preserving
formulation; synthetic-only results are not the gate.

## Reference and fixed architecture

Keep the iteration-26 `<16 query positions, 2 adjacent Q heads>` ownership,
32-position K tile, 128-wide KQ chunks, 3 splits, XOR-swizzled K LDS mapping,
V reuse, and separate partial combine unchanged. Only preserve the normalized
KQ fragment as FP32 through V accumulation instead of rounding it into the
FP16 fragment/accumulator path. The candidate remains explicitly opt-in.

The accepted object is 256 threads, 75 VGPR / 44 SGPR, zero spills/private
bytes, 27,008 B LDS. The expected FP32 fragment workspace adds 2,048 B LDS;
record actual compiler metadata before correctness execution. No achieved
occupancy or counter value may be inferred from static resource arithmetic.

## Gates

1. Build the isolated opt-in candidate; require zero VGPR/SGPR spills and no
   private/scratch memory. Record LDS, VGPR, SGPR and generated gfx906 code.
2. Compare the exact failing real-input row/stage against the immutable
   control, including absolute error and the existing tolerance.
3. Run the exact P8192/TG128 greedy prompt on control and candidate. Require all
   generated token IDs to match and no NaN/Inf. If parity fails, reject without
   timing.
4. Only after parity passes, run five interleaved isolated P8192 pairs against
   iteration 26 (200.612 ms median); require at least 5% attention-kernel
   improvement. A candidate below materiality is rejected even if correct.
5. Only an isolated winner receives the exact full P8192 request, short-context
   no-regression check, telemetry, and VRAM review. Preserve the iteration-26
   branch/path regardless of outcome.

## Decision

Pending the parity-first candidate result. Do not add more schedule, geometry,
unroll, or precision sweeps if this formulation fails; collect new causal
evidence before reopening another mechanism.
