# EXP-0263 — M12 GDN contract fix and dense-path qualification

## Hypothesis

The M12 continuation abort is caused by the chunk kernel interpreting the
runtime's multiplicative decay as log-decay. After correcting that contract,
the GDN path should remain finite across repeated 64-token chunks. The dense
FFN-down path still requires a separate greedy-output qualification.

## Candidate

The M12 chunk kernel now accumulates `logf(decay)` before applying cumulative
products. A missing barrier after the shared `system`/`intra_attention` fill
was also added. The standalone oracle and GPU benchmark now use the same
positive multiplicative-decay contract.

## Environment

- GPU: AMD Instinct MI50 / MI60-visible gfx906
- Model: `/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf`
- Build: `mi50-release`
- ROCm: 6.4.0 / LLVM 20
- HIP graphs: disabled
- Prompt: repeated `hello `, exact P64/P128/P513/P1024 boundaries

## Correctness

The host oracle and GPU chunk benchmark pass after the contract correction:

```text
m12-gdn-oracle: PASS
GPU chunk benchmark: max_output_error=1.2e-8, max_state_error=1.34e-7
```

The previous GDN-only P128 continuation abort is gone. Combined M12 P513/TG8
and M12 P1024/TG1 complete; the P1024 run reached 50.92 prefill tok/s.

At P128, debug argmax comparison against the default path produced:

| Path | First token |
| --- | ---: |
| Default baseline | 248046 (EOS) |
| Layer-major control | 248046 (EOS) |
| GDN-only | 248046 (EOS) |
| Dense FFN-down only | 271 |
| Combined GDN + dense | 271 |

The dense backend therefore changes the greedy result at P128. The isolated
P64 one-token result from EXP-0259 was insufficient qualification.

## Decision

**KEEP the GDN contract fix. REJECT dense-path promotion.** M12 remains opt-in;
the combined 52.99 tok/s result is not release-qualified. Do not enable the
dense FFN-down backend by default without a numerically equivalent replacement
or an explicitly accepted model-quality policy.

## Follow-up

Keep the package default on the validated path. If M12 promotion is revisited,
compare logits and recurrent/KV state at P1, P2, P63, P64, P65, P127, P128,
P129, P511, P512, P513, and the 1024-token capacity limit, with replay
determinism and long continuation checks.
