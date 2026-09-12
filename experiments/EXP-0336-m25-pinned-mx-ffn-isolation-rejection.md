# EXP-0336 — M25 pinned Mx MMQ recurrent-FFN isolation rejection

## Hypothesis

The pinned Mx MMQ contract may be useful for the measured recurrent FFN tail
even though applying it to every Mx projection regresses P512.

## Baseline

The staged Mx MMQ kernel selected by default at `ed92c2d`.

## Candidate

The already-ported pinned Mx MMQ kernel from
`mx-llama.cpp@2e9d29fe736969160f17476ec6f0a6298cee6966`, selected only for
the recurrent wide-prefill FFN gate, up, and down projections. Attention and
SSM-out projections stayed on the staged kernel.

## Environment

- AMD Instinct MI50 / gfx906 / Wave64
- Qwen3.8-27B-Q4_K_M, exact 512-token repeated-fox prompt
- Release build, context capacity `1024`
- explicit M25 H/I vector, `MIINFER_MX_Q8_BATCH=0`
- model allocation: `21,993,242,964 B`
- standalone projection screen used layer 35, Q4_K gate and down shapes

## Correctness

The standalone Q4_K projection outputs were finite and matched the Mx
contract oracle within `4.3e-7` for gate and `7.2e-7` for down. The complete
candidate P512 run finished normally.

## Results

Standalone B512 projection timings:

| projection | staged us | pinned us | delta |
| --- | ---: | ---: | ---: |
| layer-35 FFN gate Q4_K | 6849.433 | 8694.711 | +26.94% |
| layer-35 FFN down Q4_K | 7237.432 | 9153.109 | +26.47% |

One end-to-end P512 screen measured `2536.07 ms` / `201.89 tok/s` for the
staged control and `2839.87 ms` / `180.29 tok/s` for the FFN-only pinned
candidate (`+11.98%` latency).

## Interpretation

The pinned contract is slower at the exact dominant FFN shapes, not merely
when mixed into attention. Its complete port and FFN-only isolation both fail
to close the repaired H/I gap.

## Decision

**REJECT.** Remove the FFN-only selector and retain the pinned implementation
only as an opt-in historical differential under `MIINFER_MX_PINNED=1`.

## Follow-up

Do not spend another pass isolating components of this pinned schedule without
a new measured hypothesis. The remaining P512 gap requires a different
recurrent FFN execution contract or should remain an explicitly documented
stretch gap.
