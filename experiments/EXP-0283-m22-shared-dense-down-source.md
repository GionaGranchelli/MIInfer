# EXP-0283 — M22 shared dense FFN-down source

## Hypothesis

The dense FFN-down candidate can be combined with the accepted dense FFN
gate/up path if its canonical quantized source is reduced from one copy per
layer to one reusable layer-sized buffer.

## Candidate

`MIINFER_PREFILL_LAYER_MAJOR=1`
`MIINFER_PREFILL_DENSE_PROJECTIONS=1`
`MIINFER_PREFILL_DENSE_FFN_DOWN=1`.

The shared buffer was uploaded from the current layer before each layer/chunk
dispatch, then consumed by the existing FP16/rocBLAS down projection.

## Result

The candidate completed P512 at 38.69 tok/s with peak allocation
31,980,495,188 bytes. The accepted FFN-only candidate is 55.86 tok/s at
29,973,676,372 bytes. The shared-source candidate therefore loses 30.7% even
though it avoids the earlier dense-down OOM.

The loss comes from repeatedly copying the layer's canonical down tensor for
each of the eight prompt chunks; a single buffer cannot remain valid while the
outer schedule advances through all layers and chunks.

Raw artifact:
`results/m22-candidates/20260909-214158-2000444/`.

## Decision

REJECT. Remove the shared-source runtime path. A useful combined candidate
needs a same-resident representation or a schedule that does not recopy
weights in the hot prefill loop.
