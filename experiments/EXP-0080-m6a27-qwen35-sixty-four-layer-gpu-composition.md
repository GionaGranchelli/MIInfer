# EXP-0080 — M6-A27 Qwen3.8 sixty-four-layer GPU composition

## Question

Can the common qwen35 GPU executor compose all 64 main layers under the
accepted external correctness contract?

## Candidate

Added the remaining layers 32–63 to the existing `GpuLayerRef` executor. The
verified three recurrent layers plus one full-attention layer pattern is used
without a second executor or new kernels. Persistent recurrent state and
attention caches are allocated per layer. Added:

```text
--prefix64-external-contract
```

## Environment and command

```bash
cmake --build build/mi50-release \
  --target miinfer-m6a21-qwen35-gpu-hybrid-block -j2
build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  /tmp/m6a264-reference.YQO1ad --prefix64-external-contract
```

## Results

The 64-layer path reaches P0, but fails the established layer-output envelope
at P1:

| Position | First failing layer | Max abs | RMS | Relative RMS |
| ---: | ---: | ---: | ---: | ---: |
| P1 | L54 | 23.1531 | 0.379572 | 0.100529 |

The preceding layers remain within the `2.0` output diagnostic envelope. The
same run reports a recurrent-state diagnostic at L57/P1 (`max_abs=0.0666871`),
but the first large observable failure is L54 output. This is a composition
failure, not a passing full-model result.

## Decision

**RETEST.** Do not proceed to full generation or performance benchmarking.
The layer construction and mapping compile successfully, but L54 must be
localized before A27 can close.

## Follow-up

Use a narrow L54/P1 operation-boundary trace. The first probe should compare
the L54 input, normalization, QKV, recurrent output, gated output, attention
residual, FFN output, and final layer output. Do not reopen the already-cleared
recurrence/storage mechanics without new evidence.
