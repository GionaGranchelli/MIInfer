# EXP-0085 — M6-A27.5 Qwen3.8 P2 drift localization

## Question

Where does the deterministic P2 observable-token mismatch first appear in the
64-layer GPU composition?

## Hypothesis

The P2 mismatch is caused by accumulated early numerical/representation drift
and a small final logit margin, rather than a new L53/L54 state or gated-path
failure.

## Method

The existing non-aborting observable-contract executor was rerun against the
pinned external fixture. No kernel, tolerance, or production behavior was
changed.

```bash
build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  /tmp/m6a273-reference --prefix64-observable-contract
```

The P2 row was decoded into per-layer max absolute, RMS, and relative RMS
errors. The final observable result was also compared against EXP-0084.

## Results

### P2 layer-output error scan

| Layer | Max abs | RMS | Relative RMS |
| ---: | ---: | ---: | ---: |
| L0 | 0.000524521 | 0.000134354 | 0.000572944 |
| L1 | 0.0256538 | 0.00100365 | 0.00329474 |
| L2 | 0.0457478 | 0.00158644 | 0.00500848 |
| L3 | 0.0192070 | 0.00197273 | 0.00443329 |
| L4 | 0.0175552 | 0.00250110 | 0.00511853 |
| L5 | 0.0837669 | 0.00325595 | 0.00637025 |
| L6 | 0.111805 | 0.00451529 | 0.00817528 |
| L7 | 0.119465 | 0.00485449 | 0.00801388 |
| L8 | 0.111858 | 0.00575495 | 0.00845796 |
| L9 | 0.0891647 | 0.00655139 | 0.00925719 |
| L10 | 0.168903 | 0.00789629 | 0.0106530 |
| L11 | 0.133278 | 0.00852433 | 0.0107440 |
| L12 | 0.199535 | 0.00954580 | 0.0112876 |
| L13 | 0.157028 | 0.0103139 | 0.0114348 |
| L14 | 0.199257 | 0.0114261 | 0.0122956 |
| L15 | 0.181839 | 0.0121903 | 0.0127011 |
| L16 | 0.163383 | 0.0143045 | 0.0141304 |
| L17 | 0.169598 | 0.0154257 | 0.0148871 |
| L18 | 0.187820 | 0.0164943 | 0.0162973 |
| L19 | 0.265488 | 0.0179423 | 0.0182947 |
| L20 | 0.202087 | 0.0188599 | 0.0198810 |
| L21 | 0.0875903 | 0.0198650 | 0.0191785 |
| L22 | 0.0794721 | 0.0199400 | 0.0183042 |
| L23 | 0.0990982 | 0.0199573 | 0.0189877 |
| L24 | 0.135075 | 0.0205851 | 0.0188704 |
| L25 | 0.113037 | 0.0216724 | 0.0197684 |
| L26 | 0.124012 | 0.0229325 | 0.0204056 |
| L27 | 0.109067 | 0.0229767 | 0.0214191 |
| L28 | 0.104719 | 0.0236997 | 0.0229040 |
| L29 | 0.112681 | 0.0250111 | 0.0236917 |
| L30 | 0.109768 | 0.0269712 | 0.0251132 |
| L31 | 0.0987727 | 0.0279866 | 0.0256728 |
| L32 | 0.167442 | 0.0294500 | 0.0268540 |
| L35 | 0.459846 | 0.0349482 | 0.0294615 |
| L40 | 0.185265 | 0.0335198 | 0.0254288 |
| L45 | 0.142289 | 0.0351134 | 0.0253017 |
| L50 | 0.362057 | 0.0490149 | 0.0278217 |
| L53 | 0.276800 | 0.0672816 | 0.0279595 |
| L54 | 0.329764 | 0.0769681 | 0.0266095 |
| L58 | 0.643936 | 0.105616 | 0.0259351 |
| L62 | 1.29170 | 0.161454 | 0.0280390 |
| L63 | 1.91669 | 0.224144 | 0.0276212 |

The first nonzero discrepancy is already measurable at L0. It grows through
L1/L2, then fluctuates while the relative RMS remains bounded in the roughly
0.025–0.030 range in the later trunk. There is no new discontinuous L53/L54
event at P2.

### Observable consequence

The repeated final-logit result is the EXP-0084 P2 result:

```text
logits max_abs       0.353759
logits relative RMS  0.0218885
logits cosine        0.999773
top-5 overlap        5/5
reference rank       2 on GPU
reference margin     0.0349064
reference argmax     1318
GPU argmax           1044
```

The low reference margin explains why a bounded logit perturbation changes
the greedy decision at this position. The mismatch is deterministic and was
reproduced by the second observable-contract run.

## Interpretation

The P2 failure is an accumulated early activation/precision-envelope issue,
not evidence of a fresh L53/L54 implementation defect. L53/L54 remains
cleared by the prior operand and gated-path adjudications. The current data
does not identify one exact operation inside L0–L2 as the source; it narrows
the next investigation to the earliest representation boundary rather than
the late recurrent state machinery.

## Decision

**MEASUREMENT-ONLY / A27.5 complete.** Do not change tolerances, production
selection, or the external correctness contract. A27 remains RETEST because
teacher-forced observable agreement is 63/64.

## Follow-up

If strict external token agreement remains required, trace L0–L2 precision and
representation boundaries, starting with the first measurable L0/L1 drift.
Otherwise, adjudicate a documented functional-output contract using final
logits, margins, teacher-forced decisions, and deterministic replay. Do not
resume broad L53/L54 debugging without new evidence.
