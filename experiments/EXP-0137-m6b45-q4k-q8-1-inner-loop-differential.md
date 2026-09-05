# EXP-0137 — M6-B45 Q4_K×Q8_1 inner-loop differential

## Question

After rejecting alternate workgroup mappings, what exact execution difference
remains between MIInfer's production Q4_K×Q8_1 path and the pinned llama.cpp
gfx906 MMVQ path, and is there a non-geometric mechanism worth isolating next?

## Baseline

MIInfer commit `fbbeec9`, production B41 path:

* one 256-thread workgroup handles two independent output rows;
* Q8_1 input and decoded Q4_K metadata are staged in LDS;
* Q4 nibble payload remains in global memory;
* each lane invokes the existing two-part dot helper;
* decoded metadata and the two-row reduction are unchanged.

The fresh position-63 profile on the selected Qwen3.8-27B Q4_K_M GGUF measured:

```text
total GPU event: 73.9278 ms/token
layer sum:       70.6851 ms/token
final LM head:    2.45568 ms
allocations:      0
```

Representative Q4_K×Q8_1 stages were:

```text
recurrent FFN Gate/Up: 0.26080–0.26544 ms
recurrent FFN Down:    0.41840–0.44576 ms
full-attention Gate/Up:       0.27792 ms
full-attention Down:          0.43008 ms
```

## External comparison

The pinned llama.cpp source is commit
`c0bc8591e8815c63cb01dd3f051a8b0df02501c`.

For Q4_K×Q8_1, its `vec_dot_q4_K_q8_1` wrapper:

1. loads the two Q4 payload words;
2. unpacks the scale/minimum metadata into compact local values;
3. loads the Q8 operand words for all `QR4_K` chunks into local arrays;
4. converts each Q8 scale once into a float array;
5. calls a helper that consumes those packed local arrays.

MIInfer's current helper performs the same dot4 arithmetic and metadata
formula, but reads the Q8 words and Q8 scale from the LDS-resident block while
executing its two-part loop. MIInfer's Q4 metadata is already decoded once per
block by B41. The two implementations therefore share the quantization
contract, but do not share the same operand staging and inner-loop form.

## Controls already run

The following alternatives are closed for this path:

| Experiment | Result |
| --- | ---: |
| two independent output rows with LDS input | B35 selected |
| full Q4_K weight staging | -3.9% TG64/TG128 |
| raw metadata staging | B39 selected, then superseded by B41 |
| decoded metadata staging | B41 selected, about +3.8% over B39 |
| two-wave split-K, llama-style row mapping | -16.25% TG64, -16.20% TG128 |

These results rule out another arbitrary workgroup or split-K sweep. They do
not test the inner-loop operand-packing difference in isolation.

## Interpretation

The remaining evidence-backed Q4 hypothesis is not “use more waves.” It is:

> Preserve B41's two-row geometry and arithmetic, but test whether packing the
> Q8 words and scales into the per-lane local form used by llama.cpp improves
> the dot4 inner loop on gfx906.

This is a bounded diagnostic hypothesis because Q4_K×Q8_1 is a large repeated
family in the current profile, and the source comparison identifies a concrete
mechanism not covered by B44.

No performance benefit is inferred from the source difference alone.

## Decision

**MEASUREMENT-ONLY.** No production code changed and no new kernel candidate
was selected. The current B41 decoded-metadata path remains production.

## Next experiment

If a direct exact-shape microbenchmark can be added without changing the
production path, M6-B46 may test one opt-in inner-loop candidate that packs the
Q8 words and scales into local values while retaining B41's two-row mapping,
Q4 payload loads, decoded metadata, dot4 arithmetic, and output reduction.

Correctness must be checked before timing. A whole-token A/B is required for
any production decision. If the candidate is not clearly faster, stop this
Q4 inner-loop family rather than trying more geometry variants.
