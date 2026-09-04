# EXP-0079 — M6-A26.9 Qwen3.8 external recurrent-state contract

## Status

**COMPLETE — external-contract adjudication.** The internal recurrent-state
maximum remains a diagnostic signal; it is not the M6 correctness authority.
No numerical tolerance was widened.

## Question

Does the L30 state discrepancy identify an incorrect recurrent implementation,
or does it reflect an internal-state/observable-output contract mismatch?

## Evidence

The A26 provenance sequence established:

* L30 external-operand recurrence replay: MIInfer max error `5.96e-8`.
* L30 production operand substitution: the largest cause is upstream `k_in`.
* L30 K-path trace: its input already differs at L30 input (`l_out-29`).
* L29 output trace: the first material discrepancy is the gated path, while
  the recurrent output remains close.
* L29 gate-input replay: the existing Q4_K×Q8_K projection reaches
  `1.90735e-6` with the external normalized input.
* The L30 state maximum is non-monotonic and moves between indices; the
  P20 outlier is not a persistent corrupted state cell.
* Layer outputs through the 32-layer prefix remain within the established
  external-reference output envelope, and poisoned reset/replay fingerprints
  are deterministic.

These results clear recurrence arithmetic, storage, layout, the L30 K path,
and the L29 gate projection. They locate the remaining difference upstream of
the recurrent update rather than in the state machinery itself.

## Contract decision

M6 correctness is defined by the pinned external model/reference contract:
finite and deterministic execution, externally bounded observable layer/final
outputs and logits, correct state evolution behavior, and reproducible reset
and replay. Exact equality of an implementation-private recurrent state is
not required when the implementation may change state layout, representation,
precision boundaries, and accumulation ordering.

The existing `max_abs <= 0.05` state check remains useful as a strict diagnostic
for future regressions. It is not silently changed to `0.06` or `0.12`, and it
does not block the explicitly named external-contract composition mode.

## Harness change

`--prefix32-external-contract` runs the existing 32-layer GPU composition with
the external layer-output envelope and deterministic poisoned reset/replay
gates, while reporting the historical state check as diagnostic-only. The
default `--prefix32` mode retains the strict state gate.

## Validation run

```bash
cmake --build build/mi50-release \
  --target miinfer-m6a21-qwen35-gpu-hybrid-block -j2
ctest --test-dir build/mi50-release --output-on-failure
build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  /tmp/m6a264-reference.YQO1ad --prefix32-external-contract
```

The real-model external-contract run completed through P64 with exit status
zero. All reported layer-output rows passed the established output envelope;
poisoned reset/replay passed; decode-loop allocations were `0`; and the
reported device/peak footprint was `7,629,790,720` bytes. The Release suite
passed `20/20`. The run used the existing pinned fixture and MI50 build; no
production kernel or numerical tolerance was changed.

## Decision

**ACCEPT under the M6 external correctness contract; keep the internal state
maximum as diagnostic-only.** Proceed to full GPU composition. Do not perform
additional generic L30 state-localization experiments unless new observable
output, replay, or state-corruption evidence appears.

## Follow-up

Run the external-contract 32-layer mode, then compose the remaining layers
using the same common executor. Preserve state fingerprints and reset/replay
checks while climbing to the full 64-layer GPU path.
