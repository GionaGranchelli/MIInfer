# EXP-0260 — M12 combined dense and chunkwise prefill

## Hypothesis

The independently validated dense FFN-down backend and chunkwise GDN backend
compose without violating the recurrent state or ordered tail contracts.

## Candidate

Exact P512 with:

- MIINFER_PREFILL_LAYER_MAJOR=1
- MIINFER_PREFILL_GDN_CHUNKWISE=1
- MIINFER_PREFILL_DENSE_FFN_DOWN=1
- MIINFER_PREFILL_CHUNK=128
- HIP graphs disabled

Decode remains unchanged.

## Correctness

The combined path completed the exact 512-token prompt without a GPU fault.
Its two components had already passed independent P64 one-token greedy checks;
the combined P512 run completed the full recurrent and attention schedule.

## Results

| Path | Prefill |
| --- | ---: |
| M11-B layer-major control | 46.60 tok/s |
| GDN-only integration | 47.79 tok/s |
| Dense B128-only integration | 51.56 tok/s |
| Combined GDN + dense B128 | 52.99 tok/s |

The combined run took 9662.70 ms for 512 prompt tokens.

## Decision

KEEP as an opt-in composition experiment. The components compose and the
combined result is materially better than either independent runtime path, but
52.99 tok/s remains below the aspirational 60 tok/s gate. Do not change the
default or decode path; further gains require another independently measured
projection or memory-plan experiment.
