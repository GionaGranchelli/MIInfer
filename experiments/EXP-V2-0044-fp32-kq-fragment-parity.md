# V2-0044 — Precision-preserving KQ-to-V path

## State

REJECTED; V2-0044 stretch closed. The immutable production control remains
canonical merge `e0649c589790f0267a8f1f45b5332532a95ae13a` / checkpoint
`e68c0f20`. No candidate was timed or promoted.

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
The reference-shaped candidate then rounded tile-local weights to FP16 and
used half2 output accumulation, while the control applies FP32 online weights
directly to V. This introduced max normalized-weight error `2.46093e-5` and
gated V/output error `2.92063e-5`, just beyond the established
`2.7563e-5` envelope. This is new causal evidence for a precision-preserving
formulation; synthetic-only results are not the gate.

## Reference and fixed architecture

Keep the iteration-26 `<16 query positions, 2 adjacent Q heads>` ownership,
32-position K tile, 128-wide KQ chunks, 3 splits, XOR-swizzled K LDS mapping,
V reuse, and separate partial combine unchanged. The single variable is the
normalized-weight-to-V precision path: keep weights FP32 in LDS and carry the
V accumulator in FP32, instead of rounding weights and the running sum through
FP16. These representations are coupled because rounding the FP32 fragment
back to half2 during consumption would erase the measured correction. This is
an opt-in specialization; production continues to launch the unchanged FP16
specialization.

The plain FP32-fragment form was previously tried on the different Iteration
32 strided schedule (Iteration 33); its full-call RMS improvement was only
about 1.4%, so it was rejected without TG128 parity or timing. The exact
layer-3 attribution changes the reason to test this coupled path: FP16
fragment rounding alone crosses the measured per-row tolerance at the first
real-input boundary. Do not claim the earlier result predicts parity or
performance here.

The accepted object is 256 threads, 75 VGPR / 44 SGPR, zero spills/private
bytes, 27,008 B LDS. The FP32 fragment adds 2,048 B LDS (29,056 B expected);
FP32 output state may change VGPR use. Record actual compiler metadata before
correctness execution and require zero spills/private bytes. Verify the
default specialization remains code/resource-equivalent to saved e68. No
achieved occupancy or counter value may be inferred from static arithmetic.

## Gates

1. Build the isolated opt-in candidate; require zero VGPR/SGPR spills and no
   private/scratch memory. Record LDS, VGPR, SGPR and generated gfx906 code.
2. Compare the exact failing real-input row/stage against the immutable
   control, including absolute error and the existing tolerance.
3. Run the exact P8192/TG128 greedy prompt on control and candidate. Require all
   generated token IDs to match and no NaN/Inf. If parity fails, reject without
   timing. This is a real-model gate, not a synthetic tolerance substitute.
4. Only after parity passes, run five interleaved isolated P8192 pairs against
   iteration 26 (200.612 ms median); require at least 5% attention-kernel
   improvement. A candidate below materiality is rejected even if correct.
5. Only an isolated winner receives the exact full P8192 request, short-context
   no-regression check, telemetry, and VRAM review. Preserve the iteration-26
   branch/path regardless of outcome.

## Decision

### Production-route parity result (2026-09-29)

An initial serving run did not exercise the candidate: its toggle had only been
wired into the standalone CLI path, while HTTP serving uses
`src/prefill_v2/attention_layer.cpp`. Those token IDs are invalid candidate
evidence and are excluded. The candidate dispatch and real-input comparator
were then moved to the actual `PrefillV2AttentionLayer::forward` route. The
comparator replays the iteration-26 FP16 kernel on the same Q/K/V/gate inputs.

On the exact production API prompt (`"hello " * 8183`, `prompt_tokens=8192`,
greedy, `max_tokens=128`), iteration 26 and the candidate each naturally
stopped after 27 output tokens and produced identical IDs. However, the
layer-3/base-512 gated-attention output comparison failed the established
`2.7563e-5` absolute-error envelope across 3,145,728 values:

| Metric | Candidate vs iteration 26 |
| --- | ---: |
| Q max absolute value / RMS | 10.4236 / 1.18734 |
| Q FP16-rounding max absolute error | 0.00389862 |
| Q two-half-expansion max absolute error | 9.53674e-7 |
| Gated attention output max absolute error | 0.00908968 |
| Gated attention output max relative error | 64.3865 |
| Gated attention output RMS error | 8.0045e-5 |
| Exactly equal values | 125 / 3,145,728 |

The candidate code object compiled at 89 VGPR / 45 SGPR, zero spills/private
bytes, and 29,056 B LDS. The resource gate passed; numerical correctness did
not.

The first failing boundary measured on this candidate is gated attention output
before O projection at layer 3, base 512. This hook does not expose candidate
KQ fragments, online max/sum, normalized weights, or pre-gate V accumulation,
so the candidate's earliest internal divergence remains UNKNOWN. The earlier
iteration-28 attribution still locates the original tolerance crossing at
FP16 normalized-weight storage; it does not explain this candidate's larger
error. Token agreement for 27 outputs does not override the failed tensor
correctness gate. No P8192 performance A/B was run.

Decision: REJECT this FP32-fragment/FP32-accumulator correction and close the
stretch on immutable iteration 26 (`e68c0f20`). V2-0043 remains PRIMARY GOAL
PASS. Record the unresolved FP16 attention numerical limitation for future
frontier work; do not continue schedule/precision variants under this goal.
