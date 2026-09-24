# EXP-0385 — Validate the Exact B64 L0 Recurrent-Wide Contract

## Question

Does the exact B64 recurrent-wide route diverge from canonical scalar replay
inside L0, and if not, does the same contract fail at L1?

## EXP-0384 limitation

EXP-0384 compared only the final B64 row at each layer. This experiment uses
the existing save-wide/restore/replay/restore-wide validator on the complete
B64 batch and preserves the candidate resource selection. It does not enable
`prefill_wide_validate` and does not change routing, workspace, residency, or
MMQ selection.

## Exact candidate-resource configuration

The R route was used with `MIINFER_EXP0383_ROUTE=R`, the EXP-0374 scheduler,
and the EXP-0377 B64 residual selector. The oracle was enabled only with
`MIINFER_EXP0385_L0_ORACLE=1` and targeted layer 0, then layer 1. B64 was
`base=896,count=64`; the same run also measured qualified B128 at
`base=768,count=128`.

## Determinism and route

The fresh R run selected `248046` and reported:

```text
complete_b512=1 partial_b128=3 partial_b64=1 residual_tokens=0
scalar_layer_run_calls=16 scalar_layer_run_tokens=1024
```

The existing C/R final-token reproduction remained stable. No timing or B4
work was performed.

## L0 oracle results

### B64 headline

```text
QKV:         RMS=0             max=0
Gate:        RMS=0             max=0
Beta:        RMS=0             max=0
Decay:       RMS=0             max=0
State:       RMS=2.36730e-7    max=5.14984e-5
History:     RMS=0             max=0
Layer output: RMS=2.67579e-5   max=0.00562668
first differing row=4 (absolute position 900)
worst row=57 (absolute position 953)
row63 RMS=0 (absolute position 959)
```

### B128 headline

```text
QKV:         RMS=0             max=0
Gate:        RMS=0             max=0
Beta:        RMS=0             max=0
Decay:       RMS=0             max=0
State:       RMS=4.43358e-7    max=9.72748e-5
History:     RMS=0             max=0
Layer output: RMS=2.82531e-5   max=0.00762653
worst row=119 (absolute position 887)
```

### B64/B128 comparison

| Observable | B128 RMS | B128 Max | B64 RMS | B64 Max | B64/B128 RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| QKV | 0 | 0 | 0 | 0 | — |
| Gate | 0 | 0 | 0 | 0 | — |
| Beta | 0 | 0 | 0 | 0 | — |
| Decay | 0 | 0 | 0 | 0 | — |
| State | 4.43358e-7 | 9.72748e-5 | 2.36730e-7 | 5.14984e-5 | 0.534 |
| History | 0 | 0 | 0 | 0 | — |
| Layer output | 2.82531e-5 | 0.00762653 | 2.67579e-5 | 0.00562668 | 0.947 |

## Full B64 output row table

The complete 64×5120 output comparison is shown below. Relative RMS is
relative to the scalar row; ΔRMS is not needed because the row pattern is
sparse rather than accumulative.

| Position | Row | Max abs | RMS | Rel RMS | Cosine |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 896 | 0 | 0 | 0 | 0 | 1.000000 |
| 897 | 1 | 0 | 0 | 0 | 1.000000 |
| 898 | 2 | 0 | 0 | 0 | 1.000000 |
| 899 | 3 | 0 | 0 | 0 | 1.000000 |
| 900 | 4 | .00314903 | 9.31583e-5 | .00047935 | .9999997 |
| 901–909 | 5–13 | 0 | 0 | 0 | 1.000000 |
| 910 | 14 | 4.95911e-5 | 4.29600e-6 | 2.20730e-5 | 1.000000 |
| 911 | 15 | 8.48770e-5 | 1.23986e-5 | 6.37463e-5 | 1.000000 |
| 912–914 | 16–18 | 0 | 0 | 0 | 1.000000 |
| 915 | 19 | .00114250 | 5.26750e-5 | .00027079 | 1.000000 |
| 916–920 | 20–24 | 0 | 0 | 0 | 1.000000 |
| 921 | 25 | .00408363 | 9.63968e-5 | .00049548 | 1.000000 |
| 922–930 | 26–34 | 0 | 0 | 0 | 1.000000 |
| 931 | 35 | .00010681 | 2.50254e-5 | .00012867 | 1.000000 |
| 932–935 | 36–39 | 0 | 0 | 0 | 1.000000 |
| 936 | 40 | 1.36755e-5 | 3.32645e-6 | 1.71193e-5 | 1.000000 |
| 937 | 41 | 3.05176e-5 | 2.11574e-6 | 1.08921e-5 | 1.000000 |
| 938–943 | 42–47 | 0 | 0 | 0 | 1.000000 |
| 944 | 48 | 5.05447e-5 | 1.02729e-5 | 5.28656e-5 | 1.000000 |
| 945 | 49 | 1.33514e-5 | 2.71702e-6 | 1.39818e-5 | 1.000000 |
| 946 | 50 | .000190735 | 1.00517e-5 | 5.17299e-5 | 1.000000 |
| 947 | 51 | 2.48868e-5 | 6.32840e-6 | 3.25628e-5 | 1.000000 |
| 948 | 52 | .000107765 | 9.30346e-6 | 4.78712e-5 | 1.000000 |
| 949 | 53 | 0 | 0 | 0 | 1.000000 |
| 950 | 54 | .00105000 | 7.44044e-5 | .00038300 | 1.000000 |
| 951 | 55 | 0 | 0 | 0 | 1.000000 |
| 952 | 56 | 1.62125e-5 | 9.95483e-7 | 5.12391e-6 | 1.000000 |
| 953 | 57 | .00562668 | .000124072 | .00063844 | 1.000000 |
| 954 | 58 | 0 | 0 | 0 | 1.000000 |
| 955 | 59 | .00154972 | 5.46995e-5 | .00028152 | 1.000000 |
| 956–959 | 60–63 | 0 | 0 | 0 | 1.000000 |

The omitted rows in ranges are individually zero within the exported FP32
comparison; the nonzero rows above are the complete set.

## L1 oracle results

Because L0 cleared, the same oracle was run at L1.

```text
B64 L1:
  QKV RMS=2.40761e-8 max=9.53674e-7 row-max=38
  Gate RMS=9.82712e-7 max=1.87159e-5 row-max=59
  Beta RMS=1.65730e-8 max=1.19209e-7 row-max=59
  Decay RMS=8.15139e-9 max=1.19209e-7 row-max=59
  State RMS=7.55327e-10 max=2.68221e-7
  History RMS=4.20609e-8 max=9.53674e-7
  Output RMS=5.09174e-5 max=0.00181961 row-max=57 row63 RMS=0

B128 L1:
  QKV RMS=2.94552e-8 max=1.90735e-6
  Gate RMS=3.50645e-7 max=1.06096e-5
  Beta RMS=1.96591e-8 max=1.19209e-7
  Decay RMS=1.06745e-8 max=1.19209e-7
  State RMS=1.22105e-9 max=5.66244e-7
  History RMS=4.28437e-8 max=9.53674e-7
  Output RMS=4.99186e-5 max=0.00180054
```

L1 B64 remains in the same numerical regime as current qualified B128.

## Classification

The B64 row error is **sparse**, not immediate, monotonic, or aligned to the
second 64-row half. Rows 896–899 and 956–959 are exact; the worst row is 953.
There is no evidence of a fixed 128-row stride, stale second-half workspace,
or a state loop executing twice.

```text
Does L0 explain the EXP-0384 L1 drift? NO
First offending family: none identified
Disposition: LEARN
B64 qualified: NO
B4 authorized: NO
```

L0 and L1 both clear the current B128 numerical regime. The small recurrent
drift is therefore not localized to a discrete L0/L1 projection, state, or
output contract by this oracle.

## Exact next PRIMARY

Determine whether B64 requires a stronger semantic strategy than direct wide
execution, beginning with a representative later recurrent layer only if a
new model-boundary comparison justifies it. Do not fix B64, optimize it,
benchmark P1022, or start B4 work.

No B64 fix, performance optimization, B4 work, or new GPU math kernel was
implemented.

## Provenance

Experiment SHA: `49ee63fcfd675adc444cd9cd6259010dbefeae77`.

Graph SHA: `2010201da8ffa22e5835d139a024627a50d18da1`
(`graphify-out/graph.json` blob).

Working-tree status: clean after the experiment record and graph refresh.
