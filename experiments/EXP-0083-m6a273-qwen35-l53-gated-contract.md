# EXP-0083 — M6-A27.3 Qwen3.8 L53 gated-path contract adjudication

## Question

Does the L53 gated operation agree with the external reference when given the
same authoritative recurrent output and gate projection operands?

## Method

The reference fixture exporter was extended to capture `z-*` gate projections.
The common 64-layer executor then replayed the existing MIInfer gated path at
L53/P1 with production, mixed, and fully external operands.

```bash
cmake --build build/mi50-release \
  --target miinfer-m6a21-qwen35-gpu-hybrid-block -j2
build/mi50-release/miinfer-m6a21-qwen35-gpu-hybrid-block \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  /tmp/m6a273-reference --prefix64-l53-gated-contract
```

## Results

| Replay / comparison | Max abs | RMS | Relative RMS |
| --- | ---: | ---: | ---: |
| production recurrent vs external | 0.00111498 | 0.0000677049 | 0.0244824 |
| production gate vs external | 0.525215 | 0.0544720 | 0.0263264 |
| MIInfer operation + external operands | 4.76837e-7 | 1.59394e-8 | 3.69406e-8 |
| MIInfer operation + external recurrent | 3.64223 | 0.0475747 | 0.110257 |
| MIInfer operation + external gate | 0.539805 | 0.0130330 | 0.0302049 |
| MIInfer operation + production operands | 4.07845 | 0.0542933 | 0.125828 |

The external-operand replay matches the external gated output to floating-point
roundoff. The gated operation is therefore cleared. Its production discrepancy
is caused by upstream operand differences, not the gated formula or kernel.

The P1 layer-output scan across L32–L53 is gradual rather than a new abrupt
failure: relative RMS grows from `0.0216373` at L32 to `0.0496428` at L53, with
L53 output reaching max error `1.02168`. This is the same discrepancy entering
L54 in EXP-0081.

## Decision

**MEASUREMENT-ONLY / A27.3 complete.** No production behavior, kernel, or
tolerance changed. A27 remains RETEST: structural 64-layer composition is
achieved, but external-contract correctness is not yet closed.

## Follow-up

Stop L53/L54 gated-operation and recurrent-state-mechanics debugging. The next
decision is whether the accumulated activation envelope is acceptable under the
external contract; otherwise trace the upstream source of the gradual drift at
the layer boundary.
