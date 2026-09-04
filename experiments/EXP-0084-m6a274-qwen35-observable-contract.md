# EXP-0084 — M6-A27.4 Qwen3.8 full-model observable contract adjudication

## Question

After allowing the 64-layer GPU executor to continue past the known
intermediate diagnostic failure, do the externally observable final hidden,
final-normalized, and LM-head outputs satisfy the external contract?

## Hypothesis

The larger internal recurrent/layer-envelope errors may be reduced by final
normalization and may not change the externally observable token decisions.

## Method

The existing 64-layer GPU prefix executor was run in diagnostic-only
`--prefix64-observable-contract` mode. Intermediate layer/state failures were
reported but did not abort this adjudication run; no tolerance or production
behavior was changed.

```text
model:   /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
fixture: /tmp/m6a273-reference
reference: pinned llama.cpp c0bc8591e8815c63cb01dd3f051a8b0df02501c9
workload: hello, 64 generated positions, teacher-forced reference tokens
mode:    --prefix64-observable-contract
```

At P0, P1, P2, P4, P8, P16, P32, and P64, the run compared final hidden,
final RMSNorm, and Q6_K×Q8_K LM-head logits. It also compared argmax, top-5
overlap, reference-winner rank, margins, and the GPU-minus-reference value at
the reference winner. The complete run was repeated with the same inputs.

## Results

### Observable checkpoints

| Position | Final hidden max | Hidden rel RMS | Hidden cosine | Final norm max | Norm rel RMS | Norm cosine | Logit max | Logit rel RMS | Logit cosine | Argmax | Top-5 | Ref winner rank |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: |
| P0  | 1.56950 | 0.0407838 | 0.999189 | 0.485288 | 0.0409075 | 0.999163 | 0.459673 | 0.0182696 | 0.999834 | PASS | 4/5 | 1 |
| P1  | 3.58613 | 0.0409188 | 0.999165 | 1.19888 | 0.0411701 | 0.999152 | 0.513512 | 0.0194318 | 0.999812 | PASS | 5/5 | 1 |
| P2  | 1.91669 | 0.0276212 | 0.999624 | 0.482181 | 0.0279611 | 0.999609 | 0.353759 | 0.0218885 | 0.999773 | **FAIL** | 5/5 | 2 |
| P4  | 5.47971 | 0.0517668 | 0.998784 | 0.981064 | 0.0482197 | 0.998848 | 0.823518 | 0.0270492 | 0.999677 | PASS | 5/5 | 1 |
| P8  | 1.96623 | 0.0256320 | 0.999672 | 0.349630 | 0.0254561 | 0.999676 | 0.364311 | 0.0180252 | 0.999848 | PASS | 5/5 | 1 |
| P16 | 5.00578 | 0.0315485 | 0.999506 | 0.546639 | 0.0300764 | 0.999548 | 0.503344 | 0.0349638 | 0.999485 | PASS | 5/5 | 1 |
| P32 | 1.06240 | 0.0243714 | 0.999715 | 0.223225 | 0.0239892 | 0.999712 | 0.251686 | 0.0167736 | 0.999860 | PASS | 5/5 | 1 |
| P64 | 1.02364 | 0.0386283 | 0.999254 | 0.592276 | 0.0390735 | 0.999237 | 0.488224 | 0.0297480 | 0.999558 | PASS | 5/5 | 1 |

At P2 the reference argmax is `1318`, while GPU argmax is `1044`. The
reference winner has rank 2 on GPU; the reference margin is only `0.0349064`.
The other checkpoints agree on argmax, and P64 has logit cosine `0.999558`
with 5/5 top-5 overlap.

### Teacher-forced token decisions

| Check | Result |
| --- | ---: |
| Positions compared | 64 |
| Matching argmax decisions | 63/64 |
| First mismatch | P2: reference `1318`, GPU `1044` |
| Repeated-run behavior | identical P2 mismatch and checkpoint metrics |

Teacher forcing keeps the reference token sequence as input at every
position, so the P2 mismatch is not caused by feeding an earlier divergent
GPU token back into the model.

## Correctness

The final distributions are highly aligned, finite, and deterministic, but
the externally observable greedy decision is not identical at P2. This is a
real failure of the strict teacher-forced token contract, even though the
reference winner remains in the GPU top-2 and all later sampled checkpoints
agree.

No intermediate tolerance was changed. The known internal layer/state checks
remain diagnostic-only in this mode; the normal production path and its
selection are unchanged.

## Decision

**MEASUREMENT-ONLY / A27.4 RETEST.** Observable alignment is strong but not
exact: teacher-forced agreement is 63/64, with a deterministic P2 mismatch.
Do not close A27, claim external-reference correctness, start generation, or
run the M6 performance campaign from this path yet.

## Follow-up

Locate the earliest precision/representation drift that affects the P2 logit
margin, using the established layer checkpoints and operation contracts. Do
not resume broad L53/L54 state-mechanics debugging unless new evidence points
there. Keep the current 64-layer structural composition and production
implementation unchanged.
