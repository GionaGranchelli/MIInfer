# EXP-0290 — Persistent-token affine MMQ

## Hypothesis

The B512 projection launches one MMQ workgroup per token tile. Expanding each
weight tile once and looping over all token tiles should reduce repeated weight
reads and LDS expansion.

## Candidate

An opt-in Q4/Q5 affine MMQ kernel was added behind
`MIINFER_M23_REPACKED_PERSIST_TOKENS=1`. It kept one 64-row weight tile in LDS,
processed B128 token slabs, and accumulated each K tile into the output.

## Correctness

The focused GPU test passed for B128, B129, B256, and B512 before the model
run. No model-sized correctness claim is made because the end-to-end candidate
was immediately rejected on performance.

## Result

The exact P512 model run completed at `17.54 tok/s` versus the row-128 control
at approximately `68 tok/s`. The global partial-output traffic and the
reordered accumulation outweighed the reduced weight-tile expansion.

## Decision

**REJECT.** The experimental kernel and switch were removed; the validated
row-128/row-64 mappings remain unchanged.
