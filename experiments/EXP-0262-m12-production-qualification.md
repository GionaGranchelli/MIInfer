# EXP-0262 — M12 production qualification gate

## Hypothesis

The combined M12 layer-major prefill path can replace the default token-major
prefill without changing the first generated token or short continuation.

## Baseline

The default runtime with `MIINFER_HIP_GRAPH=0` and no M12 prefill switches.

## Candidate

```text
MIINFER_HIP_GRAPH=0
MIINFER_PREFILL_LAYER_MAJOR=1
MIINFER_PREFILL_GDN_CHUNKWISE=1
MIINFER_PREFILL_DENSE_FFN_DOWN=1
MIINFER_PREFILL_CHUNK=128
```

## Environment

- GPU: AMD Instinct MI50 / MI60-visible gfx906
- Model: `/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf`
- Build: `mi50-release`
- ROCm: 6.4.0 / LLVM 20
- Qualification base: `1a87ecc`
- Prompt: repeated `hello `, producing exact P64, P128, and P512 cases
- Generation: greedy, TG8

## Results

Prefill-only candidate runs completed at all tested boundaries:

| Prompt | Prefill | Result |
| ---: | ---: | --- |
| P64 | 1335.09 ms / 47.94 tok/s | PASS |
| P128 | 2379.19 ms / 53.80 tok/s | PASS |
| P512 | 9842.68 ms / 52.02 tok/s | PASS |

Continuation qualification did not pass:

| Prompt | Candidate result |
| ---: | --- |
| P64 + TG8 | PASS, 35.08 tok/s |
| P128 + TG8 | ABORT: `std::out_of_range`, token ID outside tokenizer vocabulary |
| P513 + TG8 | ABORT: `std::out_of_range`, token ID outside tokenizer vocabulary |

The matched default P513 + TG8 baseline completed at 32.00 tok/s prefill and
30.81 tok/s decode. No core dump was produced by the candidate failures.

## Interpretation

M12's prefill kernels survive the 64/128/512 boundaries, but the optimized
hidden state is not yet safe to hand to decode at P128 or P513. The failure is
after prefill, during the first generated-token decode/validation step. This
is a production correctness failure even though the P512 prefill-only timing
is near the M12 research result.

## Decision

**REJECT promotion.** Keep M12 opt-in and do not change the packaged default.
The path needs boundary and continuation requalification before it can become
the normal runtime path.

## Follow-up

The failure was traced to a decay-contract mismatch. The recurrent stage stores
the positive multiplicative decay, while the chunk kernel had been summing it as
if it were log-decay before applying `expf`. The first 64-token chunk could
survive with oversized values; the second chunk overflowed and produced all-NaN
logits, causing the argmax sentinel to exceed the tokenizer vocabulary.

The chunk kernel now takes `logf` of the runtime decay before accumulating it.
The standalone oracle and GPU benchmark use the same multiplicative-decay
contract. Re-run the P64/P128/P512 boundary and continuation matrix before
promotion; until that passes, M12 remains opt-in.
