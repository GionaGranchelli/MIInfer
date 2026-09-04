# EXP-0091 — M6-A27 observable numerical-equivalence closure

## Question

Are the two remaining teacher-forced argmax mismatches numerically unstable
winner flips, or evidence of an unresolved observable model error?

## Method

Ran the existing non-aborting 64-layer observable harness with the same
Qwen3.8-27B Q4_K_M GGUF and pinned llama.cpp reference fixture as the A27.9
retest. A temporary copy of the fixture added the missing P12 logits
checkpoint. No kernels, tolerances, production defaults, or execution paths
were changed. The diagnostic reports:

```text
epsilon_top = max(abs(gpu_logit - reference_logit))
              over the union of both top-5 sets
robust       = reference_margin > 2 * epsilon_top
```

## Environment

```text
model:     /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf
fixture:   /tmp/m6a273-reference-p12
reference: llama.cpp c0bc8591e8815c63cb01dd3f051a8b0df02501c9
mode:      --prefix64-observable-contract
GPU:       MI50 / gfx906
```

## Results

| Position | Reference → GPU | Top-5 | Ref rank on GPU | GPU rank on ref | Cosine | Margin | Epsilon top | Robust? |
| ---: | :--- | :---: | ---: | ---: | ---: | ---: | ---: | :---: |
| P2 | 1318 → 1044 | 5/5 | 2 | 2 | 0.999803 | 0.0349064 | 0.127861 | NO |
| P12 | 1044 → 1459 | 5/5 | 2 | 2 | 0.999443 | 0.0483742 | 0.0737877 | NO |

Top-5 sets were:

```text
P2:  reference 1318,1044,1144,599,2688
     GPU       1044,1318,1144,599,2688

P12: reference 1044,1459,599,15066,1318
     GPU       1459,1044,599,15066,1318
```

The complete teacher-forced comparison remains **62/64**. The two failures
are P2 and P12; all other positions agree. P2 and P12 both fail the robust
winner test because their reference margins are smaller than twice the
measured top-5 perturbation. Both preserve the complete top-5 set and place
the alternate winner at rank 2 in the reference distribution.

P2 observable values include logit cosine `0.999803`, relative RMS `0.0206832`,
and reference margin `0.0349064`. P12 has logit cosine `0.999443`, relative
RMS `0.0338976`, and reference margin `0.0483742`.

The first P2 layer discrepancy is L3 max `0.00255775`, below the independently
validated L3 attention max `0.00292516` and layer max `0.00516891` from
M6-A13. No new discontinuity or implementation defect was found.

## Correctness decision

The exact llama.cpp greedy trajectory is retained as a diagnostic metric, not
the MIInfer correctness authority. A27's observable contract is accepted when
the arithmetic contracts are independently validated, execution is finite and
deterministic, state/KV reset and replay are valid, final distributions remain
strongly aligned, and any argmax flip is non-robust under the measured
margin-aware bound.

## Decision

**ACCEPT / M6-A27 CLOSED.**

64-layer GPU structural composition passes. Observable numerical equivalence
passes the external margin-aware contract. The exact teacher-forced argmax
score is recorded honestly as `62/64`, with P2 and P12 classified as
low-margin, top-5-preserving diagnostic flips. No tolerance was loosened and
no production kernel change was made in this closure milestone.

## Follow-up

Proceed to M6-A28 native autoregressive GPU generation. Validate LM-head to
argmax feedback, persistent recurrent/KV state, 16/64/128-token runs,
deterministic replay, finite outputs, zero steady-state allocations, and stable
VRAM before starting M6-B1 performance benchmarking.
