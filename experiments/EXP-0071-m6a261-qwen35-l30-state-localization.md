# EXP-0071 — M6-A26.1 Qwen3.8 L30 state localization

## Status

COMPLETE — diagnostic localization. A26 remains RETEST because the strict
`max_abs <= 0.05` recurrent-state gate is unchanged.

## Question

Is the L30/P64 recurrent-state discrepancy a growing numerical drift, a
position-specific state/cache failure, or a discrepancy introduced by the
P64 L30 execution itself?

## Baseline and change

The A26 `--prefix32` GPU executor and external fixture were retained. The
diagnostic-only `--prefix32-locate` mode was added with:

* L30 state-entry measurements at P1, P2, P4, P8, P16, P32, P48, P56, P60,
  P61, P62, P63, and P64;
* P64 entry-state comparisons for L28, L29, and L30;
* L30/P64 boundary measurements for normalized input, QKV output, recurrent
  output, gated output, attention residual, post-attention norm, FFN output,
  and layer output;
* an explicit L30 state-after-P63 versus state-entry-P64 comparison.

The temporary fixture exporter change used to generate the additional
positions was restored. No production path or tolerance was changed.

## Environment and command

* model: `/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf`
* fixture: `/tmp/m6a261-qwen38-reference.Mm3R1b`
* device: MI50 / gfx906
* build: `build/mi50-release`

```bash
build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  /tmp/m6a261-qwen38-reference.Mm3R1b --prefix32-locate
```

## L30 state-entry results

`max_abs / mean_abs / RMS / relative RMS`:

| Position | Max abs | Mean abs | RMS | Relative RMS |
| -------: | -------: | -------: | -------: | -------: |
| P1  | 0.00884902 | 0.00000465133 | 0.0000353699 | 0.0225891 |
| P2  | 0.0259392  | 0.0000261473 | 0.0000940947 | 0.0251473 |
| P4  | 0.0473679  | 0.0000535250 | 0.000171705  | 0.0342309 |
| P8  | 0.0308862  | 0.0000651821 | 0.000174835  | 0.0323431 |
| P16 | 0.0342968  | 0.0000823717 | 0.000201005  | 0.0319805 |
| P32 | 0.0183711  | 0.000103291  | 0.000221221  | 0.0316397 |
| P48 | 0.0296446  | 0.000101668  | 0.000237239  | 0.0326837 |
| P56 | 0.0249020  | 0.0000953390 | 0.000214432  | 0.0298529 |
| P60 | 0.0400108  | 0.0000978158 | 0.000244341  | 0.0339262 |
| P61 | 0.0255043  | 0.0000960240 | 0.000218214  | 0.0304639 |
| P62 | 0.0306257  | 0.0000957164 | 0.000217408  | 0.0300957 |
| P63 | 0.0438641  | 0.0000950870 | 0.000233318  | 0.0326230 |
| P64 | 0.0523846  | 0.0000939012 | 0.000225921  | 0.0315860 |

The curve fluctuates rather than growing monotonically. P64 is a small final
max-absolute increase, not a discontinuous context/cache jump. The maximum
index changes with position; at P64 it is index `86909`, where the reference
is `-0.352732` and GPU value is `-0.300347`.

## Adjacent-layer P64 state entries

| Layer | Max abs | Mean abs | RMS | Relative RMS |
| ----: | -------: | -------: | -------: | -------: |
| L28 | 0.00705622 | 0.0000798629 | 0.000153496 | 0.0262670 |
| L29 | 0.00910039 | 0.0000884305 | 0.000186178 | 0.0297696 |
| L30 | 0.0523846  | 0.0000939012 | 0.000225921 | 0.0315860 |

L30 is the first adjacent recurrent state that exceeds the existing max
absolute gate at P64. The state-after-P63 comparison is identical to the
L30 P64 entry result, proving the discrepancy predates the P64 L30 update.

## L30/P64 boundary trace

| Boundary | Max abs | RMS | Relative RMS |
| -------- | -------: | -------: | -------: |
| normalized input | 0.0655256 | 0.0142922 | 0.0173480 |
| QKV | 0.150381 | 0.0325207 | 0.0339193 |
| recurrent output | 0.000563471 | 0.0000191463 | 0.0170906 |
| gated output | 0.0832915 | 0.00519666 | 0.0594522 |
| attention residual | 0.142448 | 0.0160856 | 0.0150922 |
| post-attention norm | 0.0658471 | 0.0146105 | 0.0257459 |
| FFN output | 0.0433351 | 0.00840766 | 0.0550654 |
| layer output | 0.170143 | 0.0165451 | 0.0152839 |

The P64 recurrent output is close to the external checkpoint despite the
pre-existing state discrepancy. No P64 cache/state corruption or abrupt
operation failure was exposed by this boundary trace.

## Other A26 gates observed

* all 32 layer outputs remain within the existing output envelope;
* L28–L31 P64 output checks remain finite and pass;
* poisoned reset/replay remains exact;
* steady-state decode allocations remain zero;
* state and KV fingerprints remain deterministic.

## Decision

**LOCALIZED / A26 RETEST REMAINS.** The discrepancy is bounded and
non-monotonic, with the P64 value already present after P63. The diagnostic
does not justify changing the external state tolerance or declaring A26 a
strict correctness pass. Further investigation, if required, should target
the L30 recurrent-state update history rather than P64 attention/cache
geometry.

## Follow-up

Keep the strict A26 gate. Do not compose 64 layers until the state envelope is
either corrected or explicitly approved from a broader external error
distribution.
