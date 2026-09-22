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

## Qualified B512 numerical baseline

The existing M26-C export path produced a scalar P512 state snapshot, allowing the mandatory qualified-B512-vs-scalar state comparison without the invalid M23-disabled route. At L3:

| Tensor | Max absolute | Mean absolute | RMS |
|---|---:|---:|---:|
| K | `0.215332` | `0.0100584` | `0.0139063` |
| V | `0.078125` | `0.00529292` | `0.00736426` |

The P640 partial-tail delta (`0.00195312`) is below both qualified B512 state-envelope maxima by two orders of magnitude. The scalar M26-C export has a pending-token convention different from the prefill snapshot, so its direct continuation IDs were not used as a baseline comparison. P640 candidate versus matching fallback continuation remained exact for eight tokens.

Classification: **EXPECTED_BATCH_NUMERICAL_DRIFT** for the observed P640 K/V delta, subject to causal and prefix checks. This does not promote the partial route.

## Three-way route comparison

The available state-level matrix is:

| Route | Logical count | Base | Evidence | Result |
|---|---:|---:|---|---|
| Scalar/ordered M26-C export | 512 | 0 | qualified P512 state | L3 K/V envelope above |
| Qualified batched fallback | 512 | 0 | qualified P512 state | compared against scalar |
| Qualified fallback | 128 tail | 512 | P640 state | first L3 difference at 515 |
| Partial aligned candidate | 128 tail | 512 | P640 state | `0.00195312` K/V maximum |
| Scalar/ordered | 128 | 0 or 512 | not available as a matching prefill snapshot | not run |

The exact scalar/ordered P640 cell cannot be produced by the existing M26-C
export because that export has a pending-token snapshot convention and its
route selector is invalid on this model (ROCm memory fault or `invalid M23
wide beta/decay inputs`). The scalar P512 envelope is therefore the accepted
semantic baseline, not a claim of a complete four-cell matrix.

## Per-stage and K/V localization

No intermediate tensor dump exists for the P640 run at the normalized-input,
projection, post-normalization, post-RoPE, or pre-write boundaries. The first
observable state-level boundary is the serialized L3 active K/V cache. K and V
were separated in the B512 baseline: K max `0.215332`, V max `0.078125`.
Consequently, the evidence classifies the P640 delta but does not prove a
narrower projection, normalization, RoPE, or KV-store substage. V is not
RoPE-processed, so a future stage oracle must keep its path separate from K.

The original `position 515` is absolute position `512 + 3`, the fourth token
of the B128 tail. Serialized cache offset `64` is head 0, dimension 64 in the
head-major K record (not a cache byte offset or a separate head). A periodic
per-position/per-lane error scan was not captured by the existing snapshot
format.

## Position, causal, and prefix audit

Static path audit found the following contracts in the implementation:

| Contract | Evidence |
|---|---|
| absolute token position | scalar loop passes `base + i`; full-layer chunk passes `base_position + base + i` |
| batched KV write position | `finish_prefill_attention(base_position, count)` passes the same base/count to the fused store |
| scalar KV write position | `run()` passes `position` to the fused K/V store |
| scalar causal bound | tiled and untiled attention receive `position + 1` |
| prior prefix preservation | P640 serialized prefix `[0,512)` had no large overwrite; P1664 prefix `[0,768)` was identical |

This is a source-level contract audit, not proof of every generated kernel
bound. The snapshot format does not capture per-query causal upper bounds or
attention outputs for positions 512--516, so a tensor-level causal audit is
still required before promotion. No evidence currently demonstrates a future
read or prefix overwrite.

The separate `MIINFER_EXP0368_SCALAR_ORACLE=1` production routing selector remains rejected: disabling all wide execution caused a ROCm memory fault, while reduced variants failed with `invalid M23 wide beta/decay inputs`. It is retained only as documented failed diagnostic evidence.

## Later-base qualification attempt

The same candidate/default comparison was run at P1664 (`3xB512+B128`), with
context capacity 2048 and the same model and qualified preset. Both runs
completed at exactly 1664 prompt tokens. The candidate prefill was `9,187.32
ms` (`181.12 tok/s`); the default was `21,430.93 ms` (`77.64 tok/s`).

The candidate and default matched byte-for-byte through recurrent layers 0--2
and attention layer 3. Every attention-cache prefix position `[0,768)` was
identical. The first divergence was at attention layer 7, position 768, the
next partial-tail boundary; later attention layers accumulated large K/V
differences (up to `6.30353` K and `4.94727` V in the serialized snapshots).
This is not sufficient to qualify the later base as ordinary bounded drift.
No continuation was promoted from this state, and no causal violation was
asserted from cache comparison alone.

## Stage localization status

The state snapshot localizes the first P640 observable difference to the
written L3 active K/V cache, but the qualified B512 baseline establishes that
this size of drift is already normal for the existing batched contract.
Projection, normalization, RoPE, write-input, and cache-storage boundaries
were not separately captured. P640 showed no causal-bound violation or prefix
overwrite in the tested continuation. P1664 preserves its prefix through
position 768 but diverges at that later partial-tail boundary, so the causal
and multi-boundary contract remains unqualified.

## Decision

**LEARN** — classify the measured K/V delta as expected batched numerical drift, with no correctness fix or promotion. The state oracle and P640 continuation narrow the remaining qualification work. No new math kernel, attention redesign, or unrelated optimization was implemented.

## Next PRIMARY frontier

Complete causal/prefix state qualification for P640 and explain the P1664
position-768 divergence using stage-level tensors or an equivalent contract
oracle. Until then, do not promote the partial-tail route, fix K/V numerics,
or start another attention geometry experiment.
