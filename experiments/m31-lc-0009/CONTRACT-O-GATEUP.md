# Contract O: aligned-MMQ Gate/Up prefill

This reference covers the layer-0 Q4_K Gate and Up projections used by the aligned-MMQ prefill route. It consumes the captured `attn_post_norm.f32`, `gateup_input_mx_q8.bin`, `ffn_gate.f32`, and `ffn_up.f32`, plus the canonical Q4_K tensors from the pinned GGUF.

The capture hook writes the selected prefill row immediately after `launch_mx_q8_1_mmq_quantize`. Pin `MIINFER_MX_Q8_BATCH=0`; this is the current default and selects the non-batch affine quantizer. The QSUM selector is not passed by this prefill call; its use by the one-token decode call is a separate operation and must be qualified separately.

The scalar checker validates all 160 groups of 32 input values against the captured signed Q8 codes and FP16 `d`/`sum` fields. It reproduces the kernel's FP32 scale calculation, round-to-nearest-away-from-zero Q8 conversion, and 32-lane sum reduction. It then decodes the matching Q4_K codes and 6-bit scale/minimum fields from each of the 17,408 canonical weight rows.

For each group, the represented-operand reference term is

```text
d_weight * d_input * (q4_scale * integer_dot)
- dmin_weight * q4_minimum * captured_input_sum
```

The integer dot and scale product are exact in signed 32-bit arithmetic and convert exactly to FP32 for these bounds. The FP64 reference accumulates all 160 groups. For the GPU result, the conservative bound allows six FP32 rounding steps per group, including both scale products, correction, subtraction, and accumulation:

```text
gamma_960 * sum(abs(main_term) + abs(correction_term))
+ 960 * minimum_FP32_subnormal
+ gamma_160_FP64 * sum(abs(main_term) + abs(correction_term))
```

The checker reports each row's actual error and bound for Gate and Up independently. It passes only if every quantized input code/scale matches the captured output, all outputs are finite, and every row for both projections is within its arithmetic bound. This is operation-level evidence for the captured prompt and route; it does not by itself qualify other prompts, decode, SwiGLU, recurrent state, or the full model.

Current status: checker compiled; representative real-input trace and result are pending. No Contract O pass is claimed.
