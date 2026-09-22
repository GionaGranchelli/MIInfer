# EXP-0368 — L3 partial-attention K/V and causal contract

## Question

Is the P640 partial-tail L3 K/V difference an incorrect execution contract or ordinary numerical drift already present in the qualified B512 path?

## EXP-0367 evidence

EXP-0367 localized the first candidate difference to full-attention layer 3, active position 515, element offset 64, with maximum K/V absolute error `0.00195312`. Recurrent state through L2 matched byte-for-byte and eight-token continuation IDs matched.

## P640 controlled result

The exact P640 prompt (`B512 + B128`) was run with the default fallback and with `MIINFER_EXP0366_PARTIAL_TAIL=1`, using the EXP-0367 state snapshot.

| Observation | Result |
|---|---|
| Recurrent state through L2 | byte-identical |
| First candidate divergence | L3 full-attention active K/V |
| First active position | 515 = partial-tail local index 3 |
| First element offset | 64 = head 0, dimension 64 in serialized head-major cache |
| K/V maximum absolute error | `0.00195312` |
| Final hidden | different SHA-256 fingerprints |
| Eight-token continuation | identical: `561,3841,13477,37550,33075,888,279,15217` |
| P640 default prefill | `16,028.51 ms` |
| P640 candidate prefill | `3,255.38 ms` |

The prefix comparison showed only small serialized differences before position 512 (`K` and `V` maximum `0.00048828125` in this pair); no large prefix overwrite was observed. The candidate continuation did not read an invalid future state in the tested eight-token sequence.

## Required qualified B512 numerical baseline

The mandatory scalar/ordered P512 versus qualified B512 comparison could not be completed safely with the current production CLI. The diagnostic `MIINFER_EXP0368_SCALAR_ORACLE=1` selector first attempted to disable all wide execution and faulted the GPU with a ROCm memory access fault. A reduced selector preserving wide allocations then failed before snapshot export with `invalid M23 wide beta/decay inputs`.

There is therefore no authoritative qualified B512-vs-scalar K/V envelope yet. The `0.00195312` difference is **UNCLASSIFIED**, not declared a bug or `EXPECTED_BATCH_NUMERICAL_DRIFT`.

## Stage localization status

The state snapshot localizes the first observable difference to the written L3 active K/V cache. It does not yet distinguish projection, normalization, RoPE, write input, or cache storage. V does not undergo RoPE, so K and V must be separated in the next focused harness. Causal-bound and per-stage tensor comparisons were not claimed.

## Decision

**LEARN** — no correctness fix or promotion. The candidate is not qualified, but the state oracle and P640 evidence narrow the question. No new math kernel, attention redesign, or unrelated optimization was implemented.

## Next PRIMARY frontier

Build a focused L3 B512-vs-scalar numerical baseline using existing layer/state helpers, preserving qualified allocations and avoiding the invalid M23 route. Capture projected/normalized/RoPE/write-input K and V separately, then compare the same objects for the B128 tail at base positions 0 and 512. Do not change the K/V implementation until that baseline classifies `0.00195312`.
