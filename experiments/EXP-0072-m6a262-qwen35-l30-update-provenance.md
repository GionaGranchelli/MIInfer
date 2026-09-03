# EXP-0072 — M6-A26.2 Qwen3.8 L30 update provenance

## Status

COMPLETE — diagnostic provenance. A26 remains RETEST; no tolerance or
production-selection change was made.

## Question

Which L30 recurrent update creates the P64 state discrepancy, and is tracked
state element `86909` corrupted by the GPU update or already different in the
recurrence inputs/contract?

## Diagnostic

The existing 32-layer GPU executor was run with a state-only external fixture
containing L30 `state_predelta` checkpoints at every position P0–P64. The
diagnostic tracked state index `86909`, which maps to head 5, row 38, column
125, and recorded the moving global maximum at every position.

For transitions `P3→P4`, `P7→P8`, `P31→P32`, `P59→P60`, `P60→P61`,
`P62→P63`, and `P63→P64`, it captured the previous state row, decay/key dot,
delta, and stored state element. The scalar reconstruction used the same
normalized key/query, value, beta, decay, and state-row inputs consumed by the
GPU kernel.

## Environment and command

* model: `/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf`
* fixture: `/tmp/m6a262-state-reference`
* device: MI50 / gfx906
* reference: pinned llama.cpp commit `c0bc8591e8815c63cb01dd3f051a8b0df02501c9`

```bash
build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  /tmp/m6a262-state-reference --prefix32-provenance
```

## Moving maximum and tracked-index results

| Position | Global max index | Global max abs | Error at 86909 |
| -------: | ---------------: | --------------: | -------------: |
| P1  | 86816 | 0.00884902 | 0.00202797 |
| P2  | 86900 | 0.0259392  | 0.00240872 |
| P4  | 86900 | 0.0473679  | 0.00138479 |
| P8  | 86908 | 0.0308862  | 0.0112366 |
| P16 | 86796 | 0.0342968  | 0.0187410 |
| P32 | 86796 | 0.0183711  | 0.00409675 |
| P48 | 86886 | 0.0296446  | 0.0246794 |
| P56 | 86866 | 0.0249020  | 0.00434986 |
| P60 | 86796 | 0.0400108  | 0.00251323 |
| P61 | 86811 | 0.0255043  | 0.0231740 |
| P62 | 86900 | 0.0306257  | 0.00327560 |
| P63 | 86816 | 0.0438641  | 0.00439402 |
| P64 | 86909 | 0.0523846  | 0.0523846 |

The global maximum index moves. Index `86909` is not permanently dominant,
but becomes the maximum at P64.

## Tracked update provenance

| Transition | Previous | Decay | Beta | Value | Key dot | Delta | GPU stored | Reference next | Abs error |
| ---------: | -------: | -----: | ----: | -----: | -------: | -----: | ---------: | -------------: | ---------: |
| 3→4   | 0.231201  | 0.0154515 | 0.998401 | 2.21259 | 0.0180120 | 2.19107 | -0.0132808 | -0.0118960 | 0.00138479 |
| 7→8   | -0.156907 | 0.0223623 | 0.999249 | 2.10038 | 0.0225107 | 2.07631 | -0.0608839 | -0.0496472 | 0.0112366 |
| 31→32 | -0.163203 | 0.0446343 | 0.998305 | 2.41131 | 0.0532664 | 2.35405 | 0.484946  | 0.489043  | 0.00409675 |
| 59→60 | 0.179538  | 0.181235  | 0.995472 | 2.40659 | 0.110747  | 2.28544 | -0.263971 | -0.261458 | 0.00251323 |
| 60→61 | -0.263971 | 0.160833  | 0.988338 | 2.37105 | 0.211795  | 2.13408 | -0.259924 | -0.236750 | 0.0231740 |
| 62→63 | 0.242746  | 0.103621  | 0.987087 | 2.50250 | 0.0211254 | 2.44933 | -0.412016 | -0.416410 | 0.00439402 |
| 63→64 | -0.412016 | 0.106981  | 0.995988 | 2.68022 | 0.00915641 | 2.66034 | -0.300347 | -0.352732 | 0.0523846 |

For every sampled transition, the reconstructed GPU candidate equals the
stored GPU state element. At P63→P64 specifically:

```text
GPU candidate = GPU stored = -0.300347
reference     =             -0.352732
absolute error              =  0.0523846
```

## Interpretation

This is not a state-write race, cache-layout corruption, or a mismatch between
the GPU kernel's update formula and its stored result. The decisive error is a
difference between the external reference next state and the GPU recurrence
evaluated from its current inputs. The moving maximum and non-monotonic global
curve argue against one permanently corrupted state cell, although the
P63→P64 transition is a real local jump for index `86909`.

The evidence does not yet distinguish a GPU/reference input-contract
difference from expected arithmetic sensitivity, because this fixture does
not export every reference-side beta/key/value intermediate used by the
update.

## Decision

**LOCALIZED / A26 RETEST REMAINS.** The GPU update is internally self-consistent
and the anomalous transition is identified. Do not change the `0.05` gate or
compose 64 layers until the external state contract is explicitly resolved.

## Follow-up

If another bounded check is warranted, export the reference-side L30 update
inputs for P63→P64 and compare the recurrence term-by-term. Otherwise make the
explicit external-tolerance decision and close or correct A26.
