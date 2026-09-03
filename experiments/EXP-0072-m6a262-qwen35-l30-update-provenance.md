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
| P3  | 86848 | 0.0336006  | 0.0148033 |
| P4  | 86900 | 0.0473679  | 0.00138479 |
| P5  | 86841 | 0.0452656  | 0.00564539 |
| P6  | 86891 | 0.0307440  | 0.00145477 |
| P7  | 86909 | 0.0432683  | 0.0432683 |
| P8  | 86908 | 0.0308862  | 0.0112366 |
| P9  | 86909 | 0.0306841  | 0.0306841 |
| P10 | 86816 | 0.0257390  | 0.00201802 |
| P11 | 86816 | 0.0281659  | 0.000535235 |
| P12 | 86806 | 0.0314478  | 0.00998905 |
| P13 | 86848 | 0.0308221  | 0.00925422 |
| P14 | 86908 | 0.0276558  | 0.0112588 |
| P15 | 86900 | 0.0414867  | 0.0107320 |
| P16 | 86796 | 0.0342968  | 0.0187410 |
| P17 | 86839 | 0.0303763  | 0.00431244 |
| P18 | 86908 | 0.0190966  | 0.00459433 |
| P19 | 86894 | 0.0524756  | 0.0365377 |
| P20 | 86796 | 0.110666   | 0.0201883 |
| P21 | 86830 | 0.0368600  | 0.00976811 |
| P22 | 86886 | 0.0303240  | 0.00345447 |
| P23 | 86827 | 0.0319843  | 0.00251175 |
| P24 | 86827 | 0.0252826  | 0.00335819 |
| P25 | 86800 | 0.0427408  | 0.00837243 |
| P26 | 86839 | 0.0530387  | 0.000719309 |
| P27 | 86861 | 0.0576879  | 0.00607301 |
| P28 | 86803 | 0.0580811  | 0.0169882 |
| P29 | 86830 | 0.0501156  | 0.00713657 |
| P30 | 86830 | 0.0488652  | 0.00353801 |
| P31 | 86802 | 0.0227770  | 0.00319970 |
| P32 | 86796 | 0.0183711  | 0.00409675 |
| P33 | 86796 | 0.0293306  | 0.0240124 |
| P34 | 86806 | 0.0319973  | 0.000447690 |
| P35 | 86796 | 0.0351763  | 0.00116125 |
| P36 | 86884 | 0.0305303  | 0.0130147 |
| P37 | 86796 | 0.0301481  | 0.00679611 |
| P38 | 86843 | 0.0268479  | 0.0124055 |
| P39 | 86909 | 0.0315555  | 0.0315555 |
| P40 | 86796 | 0.0273076  | 0.00334573 |
| P41 | 86843 | 0.0485338  | 0.0241293 |
| P42 | 86827 | 0.0189880  | 0.0152509 |
| P43 | 86830 | 0.0427718  | 0.00335640 |
| P44 | 86811 | 0.0256489  | 0.0220695 |
| P45 | 86884 | 0.0374337  | 0.000804625 |
| P46 | 86884 | 0.0388538  | 0.0123097 |
| P47 | 86816 | 0.0371115  | 0.0172507 |
| P48 | 86886 | 0.0296446  | 0.0246794 |
| P49 | 86843 | 0.0498800  | 0.00381558 |
| P50 | 86909 | 0.0338128  | 0.0338128 |
| P51 | 86796 | 0.0369210  | 0.00288773 |
| P52 | 86796 | 0.0268602  | 0.00659683 |
| P53 | 86886 | 0.0361774  | 0.0265415 |
| P54 | 86796 | 0.0476248  | 0.0180393 |
| P55 | 86861 | 0.0227220  | 0.00474143 |
| P56 | 86866 | 0.0249020  | 0.00434986 |
| P57 | 86894 | 0.0206592  | 0.0127777 |
| P58 | 86884 | 0.0358265  | 0.0254314 |
| P59 | 86871 | 0.0536357  | 0.0205887 |
| P60 | 86796 | 0.0400108  | 0.00251323 |
| P61 | 86811 | 0.0255043  | 0.0231740 |
| P62 | 86900 | 0.0306257  | 0.00327560 |
| P63 | 86816 | 0.0438641  | 0.00439402 |
| P64 | 86909 | 0.0523846  | 0.0523846 |

The global maximum index moves. Index `86909` is not permanently dominant,
but becomes the maximum at P64. The largest global sampled error is
`0.110666` at P20, index `86796`; the earlier sparse A26.1 sample did not
include P20.

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
curve argue against one permanently corrupted state cell. However, the full
scan found a larger P20 outlier than the earlier sparse sample, and the
P63→P64 transition remains a real local jump for index `86909`.

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
