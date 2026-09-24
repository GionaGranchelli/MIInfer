# EXP-0386 — Determine Whether Normal B64 Drift Causes L3 Semantic Amplification

## Question

Does keeping the first recurrent group L0–L2 scalar restore the P960 B64
semantic result, or do later recurrent groups also contribute materially?

## EXP-0384/0385 basis

EXP-0384 showed L2→scalar-L3 amplification: L2 last-row RMS `0.000877` and
L3 RMS `1.011864`. EXP-0385 showed L0 and L1 B64 replay remained in the
current B128 numerical regime. EXP-0386 therefore tested L2 and then the
bounded first-group scalar route.

## L2 B64 oracle

The exact R route and same-resource scalar replay were used at B64
`base=896,count=64`, with a current-head B128 comparison at `base=768,count=128`.

### B64 L2

```text
QKV:         RMS=3.69523e-8  max=1.43051e-6  row-max=49
Gate:        RMS=2.38550e-6  max=4.39882e-5  row-max=15
Beta:        RMS=1.98628e-8  max=1.19209e-7  row-max=49
Decay:       RMS=1.88824e-8  max=2.08616e-7  row-max=24
State:       RMS=4.79063e-10 max=8.94070e-8
History:     RMS=0          max=0
Output:      RMS=7.67951e-5  max=0.00402641
Worst row: 14
Row63 RMS: 0
```

### B128 L2

```text
QKV:         RMS=3.54563e-6  max=1.56282e-4  row-max=107
Gate:        RMS=8.64048e-6  max=2.58774e-4  row-max=107
Beta:        RMS=1.98139e-8  max=1.19209e-7
Decay:       RMS=1.72791e-8  max=2.08616e-7
State:       RMS=1.97839e-9  max=3.72529e-7
History:     RMS=2.77686e-8  max=4.76837e-7
Output:      RMS=1.37124e-4  max=0.00388169
Worst row: 108
Row63 RMS: 7.47891e-5
```

| Observable | B128 RMS | B128 Max | B64 RMS | B64 Max | B64/B128 RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| QKV | 3.54563e-6 | 1.56282e-4 | 3.69523e-8 | 1.43051e-6 | 0.0104 |
| Gate | 8.64048e-6 | 2.58774e-4 | 2.38550e-6 | 4.39882e-5 | 0.276 |
| Beta | 1.98139e-8 | 1.19209e-7 | 1.98628e-8 | 1.19209e-7 | 1.002 |
| Decay | 1.72791e-8 | 2.08616e-7 | 1.88824e-8 | 2.08616e-7 | 1.093 |
| State | 1.97839e-9 | 3.72529e-7 | 4.79063e-10 | 8.94070e-8 | 0.242 |
| History | 2.77686e-8 | 4.76837e-7 | 0 | 0 | 0 |
| Output | 1.37124e-4 | 0.00388169 | 7.67951e-5 | 0.00402641 | 0.560 |

L2 remains within the same qualitative numerical regime as current qualified
B128. It is not an L2 contract failure.

## First-group scalar route

The diagnostic selector was:

```text
MIINFER_EXP0386_ROUTE=FIRST_GROUP_SCALAR
```

For B64 only, L0–L2 recurrent layers were scalar; L4 onward recurrent
layers were wide; all 16 attention layers remained scalar. B512 and all
three B128 chunks were unchanged.

Route proof:

```text
complete_b512=1
partial_b128=3
partial_b64=1
residual_tokens=0
scalar_layer_run_calls=19
scalar_layer_run_tokens=1216
```

This equals 3 scalar recurrent layers + 16 scalar attention layers.

## Model-boundary comparison

| Route | Token | Top-1 margin | Top-5 | Top-10 |
| --- | ---: | ---: | --- | --- |
| C | 561 | 1.114969 | 561, 16, 17, 95726, 98832 | 561, 16, 17, 95726, 98832, 220, 95772, 18, 20, 19 |
| R | 248046 | 1.678399 | 248046, 561, 271, 17, 16 | 248046, 561, 271, 17, 16, 248044, 15, 248045, 198, 1358 |
| S | 16 | 0.595522 | 16, 17, 248046, 561, 220 | 16, 17, 248046, 561, 220, 18, 19, 20, 15, 119073 |

| Pair | Hidden RMS | Hidden cosine | Final-norm RMS | Final-norm cosine | Logit RMS | Logit cosine | Top-10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C vs S | 5.277277 | 0.901557 | 0.862713 | 0.897702 | 0.982699 | 0.979132 | 7/10 |
| C vs R | 6.173692 | 0.682461 | 1.560446 | 0.666410 | 1.577133 | 0.941810 | 3/10 |
| S vs R | 6.769040 | 0.771662 | 1.321434 | 0.759105 | 1.599066 | 0.948358 | 5/10 |

Token ranks/logits:

```text
       token 561             token 248046
C      rank 1 / 11.816545    rank 18 / 7.935751
R      rank 2 / 13.464437    rank 1  / 15.142837
S      rank 4 / 9.901530     rank 3  / 10.462337
```

## Interpretation

L2 does not show a qualitatively worse B64 numerical regime than B128.
Keeping L0–L2 scalar improves the model boundary substantially, but it does
not restore token `561`; S selects token `16`. Thus the first recurrent group
is sensitive, but later recurrent-wide groups also contribute to the semantic
trajectory. The L3 scalar attention remains an amplifier, not a route defect.

```text
Does keeping L0-L2 scalar restore semantics? NO
Classification: DISTRIBUTED_RECURRENT_SENSITIVITY
B64 qualified: NO
B4 authorized: NO
```

## Decision

**LEARN.** Normal local B64 drift at L0–L2 is sufficient to alter the
trajectory, but scalarizing only the first group is not sufficient to recover
the control token. The issue is distributed recurrent sensitivity rather than
one L2 implementation failure.

## Exact next PRIMARY

Build a recurrent-group semantic sensitivity map using group-level prefixes or
selective scalar groups. Do not inspect individual kernels yet, benchmark
P1022, promote B64, or start B4 work.

No B64 production fix, performance optimization, B4 work, or new GPU math
kernel was implemented.

## Provenance

Experiment SHA: `074aec4dfb7e779c3792d8588ec2ef700fe399d8`.

Graph SHA: `4fbf1215599892dcf4b47e779ce8dab4e602974e`
(`graphify-out/graph.json` blob).

Working-tree status: clean after the experiment record and graph refresh.
