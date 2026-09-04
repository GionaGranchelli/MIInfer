# EXP-0090 — M6-A27.9 full observable-contract retest

## Question

Does the Q5_K × Q8_K integer-accumulation fix restore the previously failing
64-layer external observable contract and teacher-forced trajectory?

## Method

Reran the existing non-aborting `--prefix64-observable-contract` harness with
the same model, fixture, checkpoints, thresholds, and reference token
sequence as EXP-0084. No tolerance or diagnostic gate was changed. The run
covered the complete 64-layer composition and teacher-forced positions 0–63.

```bash
build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  /tmp/m6a273-reference --prefix64-observable-contract
```

## Key before/after results

| Position | Metric | EXP-0084 before fix | A27.9 retest |
| ---: | --- | ---: | ---: |
| P2 | final hidden max abs | 1.91669 | 1.85683 |
| P2 | final norm max abs | 0.488164 | 0.488164 |
| P2 | logit max abs | 0.353759 | 0.338049 |
| P2 | logit cosine | 0.999773 | 0.999803 |
| P2 | top-5 overlap | 5/5 | 5/5 |
| P2 | reference winner rank | 2 | 2 |
| P2 | reference margin | 0.0349064 | 0.0349064 |
| P2 | argmax | 1318 → 1044 | 1318 → 1044 |

After the fix, the P2 layer table shows L0–L2 at roundoff-level error; the
first visible P2 layer-output discrepancy moves to L3 (`0.00255775`). This
confirms the L0 projection fix landed, but it does not restore the P2 token
decision.

Representative post-fix observable results:

| Position | Final hidden max | Hidden cosine | Logit max | Logit cosine | Argmax | Top-5 |
| ---: | ---: | ---: | ---: | ---: | :---: | ---: |
| P0 | 1.74019 | 0.999127 | 0.466257 | 0.999831 | PASS | 5/5 |
| P1 | 3.99594 | 0.999471 | 0.353185 | 0.999866 | PASS | 5/5 |
| P2 | 1.85683 | 0.999703 | 0.338049 | 0.999803 | FAIL | 5/5 |
| P4 | 0.984350 | 0.999621 | 0.385915 | 0.999835 | PASS | 5/5 |
| P8 | 2.78860 | 0.999678 | 0.361026 | 0.999851 | PASS | 5/5 |
| P16 | 3.35583 | 0.999618 | 0.339232 | 0.999619 | PASS | 5/5 |
| P32 | 1.21947 | 0.999838 | 0.201268 | 0.999911 | PASS | 5/5 |
| P64 | 1.89806 | 0.999282 | 0.547389 | 0.999572 | PASS | 5/5 |

## Teacher-forced trajectory

The retest found 62/64 matching argmax decisions:

```text
P2:  reference 1318, GPU 1044
P12: reference 1044, GPU 1459
```

All other tested positions matched. The P2 mismatch remains deterministic;
the P12 mismatch is a newly exposed low-margin decision after changing the
Q5_K arithmetic path and requires no interpretation beyond this retest.

## Other checks

The run remained finite and deterministic. Existing state/KV diagnostics
remain bounded as previously observed, with later diagnostic state-envelope
warnings but no executor abort. Release CTest remains 20/20 from the same
post-fix build.

## Decision

**RETEST / A27 not closed.** The Q5_K × Q8_K fix is locally validated and
removes the L0–L2 P2 drift, but the full external teacher-forced contract is
still not exact: 62/64 decisions match. Do not start generation or the M6
performance campaign yet.

## Follow-up

If another correctness diagnostic is justified, focus on the newly exposed L3
full-attention boundary at P2. Do not reopen the cleared L0 Q5_K/Q8_K path.
