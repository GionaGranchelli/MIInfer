# EXP-0383 — Localize the First Composed-B64 Semantic Divergence

## Question

Does the P960 first-token failure come from B64 recurrent-wide composition,
B64 batched attention, or post-prefill token selection?

## EXP-0382 failure basis

EXP-0382 proved the fully composed route has zero scalar work but selected
token `248046` instead of the scalar-control token `561`. This experiment uses
the same exact P960 prompt and splits only the B64 recurrent and attention
composition families. No kernel, scheduler, or production route was changed.

## Route proofs

All routes retained `complete_b512=1`, `partial_b128=3`, and
`partial_b64=1`, with `residual_tokens=0`.

| Route | B64 recurrent | B64 attention | Scalar calls | Scalar tokens | First token |
| --- | --- | --- | ---: | ---: | ---: |
| C | scalar | scalar | 64 | 4096 | 561 |
| R | wide | scalar | 16 | 1024 | 248046 |
| F | wide | batched | 0 | 0 | 248046 |

The B512 and all three B128 chunks were unchanged across the diagnostic
selectors. `R` and `F` were selected with the test-only
`MIINFER_EXP0383_ROUTE=R/F`; `C` used the existing scalar remainder behavior.

## Determinism check

**PASS.** Two clean C runs produced identical SHA-256 fingerprints for final
hidden, final norm, and logits, and both selected `561`:

```text
C1/C2 hidden:     40428d0d55729d1fad140b0d9db4fe6b9f2f89957518d1dd941fdb580ed01fea
C1/C2 final norm: 1c23687586556e061e3f3c34fa5406398737604415faba2a34c46c667fb076d5
C1/C2 logits:     cca9423077af79b257199a4cd035502d47726486f4b6e6f7a3e68f9def8e71c4
```

Final norm and logits were finite for C, R, and F.

## Model-boundary exports

Exports were captured for final hidden, final normalized hidden, and full
logits. Pairwise metrics are max absolute, mean absolute, RMS, relative RMS
to the first route, cosine, and first differing index.

### Final hidden

| Pair | Max | Mean | RMS | Relative RMS | Cosine | First index |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C vs R | 50.768692 | 4.721450 | 6.173692 | 0.879860 | 0.682461 | 0 |
| C vs F | 51.942932 | 4.854464 | 6.335225 | 0.902882 | 0.684451 | 0 |
| R vs F | 10.627121 | 0.490512 | 0.714006 | 0.086805 | 0.997074 | 0 |

### Final norm

| Pair | Max | Mean | RMS | Relative RMS | Cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| C vs R | 15.597725 | 1.169963 | 1.560446 | 0.815400 | 0.666409 |
| C vs F | 16.394054 | 1.168865 | 1.553444 | 0.811742 | 0.668607 |
| R vs F | 1.728156 | 0.105326 | 0.148424 | 0.077828 | 0.996967 |

### Logits

| Pair | Max | Mean | RMS | Relative RMS | Cosine | Top-5 overlap | Top-10 overlap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C vs R | 8.754416 | 1.244173 | 1.577133 | 0.360713 | 0.941810 | 3/5 | 3/10 |
| C vs F | 8.814878 | 1.259073 | 1.593438 | 0.364442 | 0.943685 | 3/5 | 3/10 |
| R vs F | 1.002701 | 0.146871 | 0.183162 | 0.039054 | 0.999496 | 5/5 | 10/10 |

## Token selection boundary

The divergence is already present in final hidden and remains in final norm
and logits. It is not an argmax or token-read contract.

| Route | Argmax | Top-1 margin | Top-5 | Top-10 |
| --- | ---: | ---: | --- | --- |
| C | 561 | 1.114969 | 561, 16, 17, 95726, 98832 | 561, 16, 17, 95726, 98832, 220, 95772, 18, 20, 19 |
| R | 248046 | 1.678399 | 248046, 561, 271, 17, 16 | 248046, 561, 271, 17, 16, 248044, 15, 248045, 198, 1358 |
| F | 248046 | 0.924194 | 248046, 561, 271, 17, 16 | 248046, 561, 271, 17, 16, 248044, 15, 198, 1358, 248045 |

| Route | Token 561 rank / logit | Token 248046 rank / logit |
| --- | --- | --- |
| C | 1 / 11.816545 | 18 / 7.935751 |
| R | 2 / 13.464437 | 1 / 15.142837 |
| F | 2 / 13.706633 | 1 / 14.630826 |

## Family classification

**Outcome B — B64 recurrent-wide semantic contract.** R already matches F's
wrong token and shows the material C→R final-hidden/logit divergence, while
R→F is substantially smaller and preserves identical top-5/top-10 ordering.
The first material model-boundary effect is introduced by B64 recurrent-wide
composition. B64 attention is not the next target.

No bounded layer localization was run: the family split already identifies
the next PRIMARY, and the protocol requires stopping at that boundary.

## Decision

**LEARN.** The composed-B64 failure is deterministic and originates before
B64 batched attention is required. The wrong token is not caused by final
normalization, LM-head argmax, or token readout.

```text
B64 qualified: NO
B4 authorized: NO
```

## Exact next PRIMARY

**B64 recurrent-wide semantic contract.** The next experiment may use only a
coarse recurrent layer-boundary map (L0/L1/L2/L15/L31/L47/L63, with binary
search if needed) to identify the first material amplification boundary.

No performance optimization, production B64 promotion, B4 work, or new GPU
math kernel was implemented.

## Provenance

Experiment SHA: `5f655709f4e6671a6d75247ddce5294b6103ea27`.

Graph SHA: `8ccd0c4fdd3cde6f23a990da528cd4483873c348`
(`graphify-out/graph.json` blob).

Working-tree status: clean after the experiment record and graph refresh.
