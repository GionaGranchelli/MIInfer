# EXP-0073 — M6-A26.3 Qwen3.8 recurrent-state contract adjudication

## Status

COMPLETE — external-input recurrence replay. A26 remains RETEST; no tolerance
or production-selection change was made.

## Question

Does the L30 recurrent update disagree with the external contract when both
implementations receive the same authoritative P19 inputs?

## Reference capture

The pinned llama.cpp reference was temporarily run with autoregressive fused
GDN disabled so its mathematical `q_in`, `k_in`, `v_in`, `b_in`, and `g_in`
nodes were available to the evaluation callback. Only L30 at P19 and P20 was
captured, together with `state_predelta-30`. The llama.cpp source and the
normal MIInfer exporter were restored after capture.

Reference commit:
`c0bc8591e8815c63cb01dd3f051a8b0df02501c9`

The external recurrence repeats the 16 key heads across 48 value heads with
`key_head = value_head % 16`, applies `decay = exp(g)`, and updates each state
row with the external beta/key/value operands.

## Environment and command

* model: `/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf`
* device: MI50 / gfx906
* build: `build/mi50-release`
* fixture: temporary `/tmp/m6a263-reference.*`

```bash
build/mi50-release/miinfer-m6a263-qwen35-recurrent-contract \
  /tmp/m6a263-reference.lzFw8Z
```

## Results

| Inputs | Recurrence | Max abs | RMS |
| --- | --- | ---: | ---: |
| External | External replay | `2.98023e-08` | `2.02857e-10` |
| External | MIInfer GPU | `5.96046e-08` | `2.49816e-10` |

For tracked index `86796`:

```text
external P20 state = 0.361484
external replay     = 0.361484
MIInfer GPU replay  = 0.361484
```

## Interpretation

The MIInfer GPU recurrence agrees with the external recurrence when given the
same external state and update operands. This clears the recurrence kernel's
formula, state layout, and update ordering as the source of the observed
L30/P20 discrepancy.

The remaining discrepancy is therefore upstream of this recurrence replay:
the state/input representation entering the production GPU update differs
from the external contract, or the checkpoint contract is not the same
boundary. This experiment does not adjudicate that upstream representation.

## Decision

**KEEP / A26.3 complete.** Stop investigating GPU state storage and recurrence
mechanics. Do not change the `0.05` gate and do not compose 64 layers yet.
A26 requires an explicit decision about the external state checkpoint
contract before closure.

## Follow-up

If A26 must be closed under a state-numerical contract, compare the production
GPU L30 operands against the external P19 operands at the input boundary.
Otherwise define a justified observable-state contract using downstream
recurrent output and long-position external behavior; do not replace the
existing threshold merely to make the test pass.
