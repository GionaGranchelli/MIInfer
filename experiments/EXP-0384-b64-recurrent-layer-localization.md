# EXP-0384 — Locate the First Material B64 Recurrent-Wide Layer Divergence

## Question

At which recurrent layer boundary does the exact P960 B64 recurrent-wide route
first show material amplification relative to scalar control?

## EXP-0383 basis

EXP-0383 showed that C→R divergence is introduced by B64 recurrent-wide
composition. C selects `561`; R selects `248046`. R→F is close, so attention
is not in scope here.

## C/R route proofs

Both fresh captures used P960 = B512 + 3×B128 + B64, with
`complete_b512=1`, `partial_b128=3`, `partial_b64=1`, and `residual_tokens=0`.

```text
C: scalar B64 recurrent, scalar B64 attention, scalar calls/tokens=64/4096, token=561
R: wide B64 recurrent, scalar B64 attention, scalar calls/tokens=16/1024, token=248046
```

B512 and all three B128 chunks were unchanged by the diagnostic selector.

## Determinism

**PASS.** Fresh C and R reproductions retained the EXP-0383 tokens and stable
final-hidden fingerprints. The C layer export is 64 × 5120 FP32 values and
the R export has the same shape. The capture was limited to B64 base 896,
count 64, last token position 959, after each completed layer handoff.

## Layer-output contract

The exported row is the output for absolute prompt position 959 after each
layer. Layers `L3, L7, ..., L63` are scalar attention in both routes; all
other layers are recurrent. Metrics are C vs R, with relative RMS measured
against the C row and ΔRMS measured against the preceding layer's RMS.

## Full C-vs-R layer table

| Layer | Kind | Max abs | RMS | Rel RMS | Cosine | ΔRMS |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| 0 | recurrent | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 |
| 1 | recurrent | 0.001892 | 0.000264 | 0.001089 | 0.999999 | 0.000264 |
| 2 | recurrent | 0.029448 | 0.000877 | 0.003762 | 0.999995 | 0.000613 |
| 3 | attention-scalar | 69.671852 | 1.011864 | 2.700934 | 0.980814 | 1.010987 |
| 4 | recurrent | 68.550148 | 0.997189 | 2.495313 | 0.979627 | -0.014675 |
| 5 | recurrent | 66.322945 | 0.967197 | 2.214496 | 0.979711 | -0.029992 |
| 6 | recurrent | 63.577110 | 0.930075 | 2.083841 | 0.974618 | -0.037121 |
| 7 | attention-scalar | 76.848808 | 1.119250 | 2.089226 | 0.971463 | 0.189175 |
| 8 | recurrent | 72.901749 | 1.069764 | 1.759963 | 0.963924 | -0.049486 |
| 9 | recurrent | 71.423386 | 1.051635 | 1.634087 | 0.960552 | -0.018129 |
| 10 | recurrent | 69.826385 | 1.032516 | 1.556620 | 0.955031 | -0.019119 |
| 11 | attention-scalar | 81.949417 | 1.206529 | 1.763683 | 0.948763 | 0.174013 |
| 12 | recurrent | 80.104019 | 1.185691 | 1.650519 | 0.942606 | -0.020838 |
| 13 | recurrent | 77.988403 | 1.161011 | 1.537764 | 0.937265 | -0.024680 |
| 14 | recurrent | 74.556870 | 1.123632 | 1.410625 | 0.926310 | -0.037378 |
| 15 | attention-scalar | 72.029755 | 1.103965 | 1.331849 | 0.922018 | -0.019668 |
| 16 | recurrent | 69.679771 | 1.079498 | 1.256016 | 0.916465 | -0.024467 |
| 17 | recurrent | 68.381889 | 1.069608 | 1.241460 | 0.905812 | -0.009890 |
| 18 | recurrent | 59.973000 | 0.985522 | 1.126600 | 0.873027 | -0.084086 |
| 19 | attention-scalar | 59.175026 | 1.012137 | 1.197408 | 0.799041 | 0.026615 |
| 20 | recurrent | 48.224613 | 0.906926 | 1.065164 | 0.769518 | -0.105211 |
| 21 | recurrent | 35.694351 | 0.802424 | 0.834079 | 0.801500 | -0.104502 |
| 22 | recurrent | 26.919868 | 0.747621 | 0.699191 | 0.823526 | -0.054803 |
| 23 | attention-scalar | 25.718727 | 0.734661 | 0.707795 | 0.814714 | -0.012960 |
| 24 | recurrent | 25.034142 | 0.743967 | 0.720647 | 0.800001 | 0.009306 |
| 25 | recurrent | 27.563515 | 0.797830 | 0.757671 | 0.774181 | 0.053863 |
| 26 | recurrent | 27.126923 | 0.842969 | 0.780714 | 0.739847 | 0.045139 |
| 27 | attention-scalar | 28.300766 | 0.882819 | 0.850924 | 0.688933 | 0.039850 |
| 28 | recurrent | 28.854370 | 0.883687 | 0.859081 | 0.687923 | 0.000868 |
| 29 | recurrent | 28.897396 | 0.890965 | 0.870308 | 0.673652 | 0.007278 |
| 30 | recurrent | 19.894085 | 0.854621 | 0.795996 | 0.691693 | -0.036344 |
| 31 | attention-scalar | 21.820030 | 0.918988 | 0.854851 | 0.643506 | 0.064367 |
| 32 | recurrent | 26.097141 | 0.973523 | 0.905808 | 0.593373 | 0.054535 |
| 33 | recurrent | 27.230110 | 0.951767 | 0.904044 | 0.596798 | -0.021756 |
| 34 | recurrent | 21.152973 | 0.958293 | 0.888885 | 0.573618 | 0.006525 |
| 35 | attention-scalar | 21.573519 | 0.989310 | 0.928436 | 0.510094 | 0.031018 |
| 36 | recurrent | 21.581203 | 1.012491 | 0.911093 | 0.546585 | 0.023181 |
| 37 | recurrent | 27.458908 | 1.068289 | 0.944790 | 0.552271 | 0.055798 |
| 38 | recurrent | 25.754765 | 1.069524 | 0.903091 | 0.597902 | 0.001235 |
| 39 | attention-scalar | 26.685371 | 1.051102 | 0.924859 | 0.588024 | -0.018422 |
| 40 | recurrent | 24.734642 | 1.076170 | 0.910425 | 0.588453 | 0.025068 |
| 41 | recurrent | 30.916580 | 1.154367 | 1.011559 | 0.543896 | 0.078197 |
| 42 | recurrent | 27.232689 | 1.227236 | 1.038334 | 0.509508 | 0.072869 |
| 43 | attention-scalar | 34.203957 | 1.630811 | 1.410089 | 0.401316 | 0.403575 |
| 44 | recurrent | 45.595085 | 1.692966 | 1.469588 | 0.393661 | 0.062155 |
| 45 | recurrent | 45.235935 | 1.668051 | 1.413824 | 0.422141 | -0.024915 |
| 46 | recurrent | 44.342266 | 1.659726 | 1.369511 | 0.394689 | -0.008326 |
| 47 | attention-scalar | 38.467300 | 1.721376 | 1.350898 | 0.407435 | 0.061651 |
| 48 | recurrent | 46.019691 | 1.775389 | 1.354265 | 0.392110 | 0.054013 |
| 49 | recurrent | 41.077621 | 1.741003 | 1.322472 | 0.376046 | -0.034386 |
| 50 | recurrent | 57.063736 | 1.907191 | 1.263008 | 0.399735 | 0.166187 |
| 51 | attention-scalar | 30.428137 | 1.843310 | 1.137142 | 0.406292 | -0.063880 |
| 52 | recurrent | 17.953995 | 1.958583 | 1.067717 | 0.426241 | 0.115272 |
| 53 | recurrent | 19.586386 | 2.110822 | 1.011685 | 0.450137 | 0.152239 |
| 54 | recurrent | 94.708702 | 2.616664 | 1.031475 | 0.614090 | 0.505841 |
| 55 | attention-scalar | 56.168724 | 2.432413 | 0.907522 | 0.634468 | -0.184250 |
| 56 | recurrent | 47.284668 | 2.485178 | 0.867700 | 0.645747 | 0.052765 |
| 57 | recurrent | 43.695084 | 2.516192 | 0.836226 | 0.668511 | 0.031013 |
| 58 | recurrent | 73.854507 | 2.716305 | 0.785201 | 0.744846 | 0.200114 |
| 59 | attention-scalar | 25.095154 | 2.741824 | 0.851109 | 0.635143 | 0.025518 |
| 60 | recurrent | 25.051594 | 2.909949 | 0.797903 | 0.667921 | 0.168125 |
| 61 | recurrent | 24.369465 | 3.175339 | 0.830188 | 0.651607 | 0.265390 |
| 62 | recurrent | 33.806992 | 3.772468 | 0.787055 | 0.662605 | 0.597129 |
| 63 | attention-scalar | 50.768692 | 6.173692 | 0.879860 | 0.682461 | 2.401224 |

## First difference versus first material amplification

The first nonzero difference is at **L1**, with RMS `0.000264`; L2 remains
small at RMS `0.000877`. L3, which is scalar attention in both routes, rises
to RMS `1.011864` and cosine `0.980814`. This is an attention sensitivity
amplifier, not evidence of an attention route defect: its input already
differs, and attention execution is identical in C and R.

After L3, the error persists and evolves through recurrent boundaries. There
is no isolated recurrent boundary with a sharp first material jump; later
increases at L43, L54, L62, and L63 are downstream amplification. The correct
classification is:

```text
SMOOTH_RECURRENT_ACCUMULATION
first nonzero recurrent boundary: L1
first large model-boundary amplification: scalar L3 sensitivity to changed L2 input
```

No optional state/history fingerprint or prefix substitution was needed.
No internal stage attribution was performed.

## Decision

**LEARN.** The direct layer map does not justify naming a single defective
recurrent layer. B64 recurrent-wide drift begins at L1, remains small through
L2, and is strongly amplified by the unchanged scalar L3 attention boundary.

```text
B64 qualified: NO
B4 authorized: NO
```

## Exact next PRIMARY

Representative B64 recurrent-wide numerical contract, beginning with L0 and
comparing against the accepted B128 wide numerical behavior. EXP-0385 may use
the existing scalar recurrent oracle for targeted internal replay; it must not
reopen attention or perform broad tensor archaeology.

No defect fix, performance optimization, production B64 promotion, B4 work, or
new GPU math kernel was implemented.

## Provenance

Experiment SHA: `7723967db620fc38d9d8ffc7755f2f70ade6ad6d`.

Graph SHA: `a0fa1728f1e6134c82b1632eb7c1a949d54e269f`
(`graphify-out/graph.json` blob).

Working-tree status: clean after the experiment record and graph refresh.
