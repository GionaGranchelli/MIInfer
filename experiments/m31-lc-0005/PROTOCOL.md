# M31-LC-0005 — First Route-Divergence Isolation

## Objective

Under the frozen M31-LC-0002 workload, find the earliest token position,
model layer, and operation where fused and MMQ-only routes differ. At that
location, retain the common input, each production Q8 activation buffer, and
each route's output. Separately decompose the layer-0 Q6_K QKV result into
activation quantization error and kernel arithmetic error.

This is a correctness attribution task. It does not select or implement a new
performance `PRIMARY`; the active performance frontier in
`docs/performance-research-protocol.md` remains untouched.

## Frozen workload and environment

Use the exact M31-LC-0002 prompt and model:

- Model SHA-256: `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.
- Prompt IDs SHA-256: `5acd37f00d9e670f4b4e06d46fa12349e1dee06e356e6e73a186d74672f5bb47`.
- Forced history: `[220, 248046, 198, 248045]`.
- Context capacity: 2,304; prefill tile: 512; greedy sampling.
- Source base: M31-LC-0004 commit `bbbe9b8b7ec9cd439bde2c050da0ac40cdeeffb6`.
- GPU: Z840 MI50, BDF `0000:06:00.0`, unique ID `0x21678e17348c2f7`.
- OCI image: `localhost/miinfer-dev:rocm-7.2.1`, digest
  `sha256:bdc5ed42c985a6a333083e023225f248825a6e1511f38b6b50fbc7d5ace3fe9e`.
- Model and prompt hashes, source identity, image digest, BDF, clocks,
  temperature, and KFD idle state must be checked and recorded for each pair.

Use the established guarded rootless Podman launch with
`--security-opt=label=disable`; this is required by the prior `/dev/kfd` SELinux
denial evidence. Do not use sudo or change host SELinux policy.

## Route comparison

Run fused then MMQ-only from fresh process/model state, with identical inputs
and the same forced output history. Capture the first generated decode token
and layer-0 recurrent FFN stages, since M31-LC-0004 established that its
position-0 prefill capture is insufficient to explain the later logit split.
If these first-token tensors remain equal, advance only through the already
frozen four-token history, capturing each token position and layer until the
first difference is found. Do not change prompt length or add a context ladder.

At the first different FFN Gate/Up site, capture:

1. The F32 post-normalized input before quantization.
2. The raw production activation quantization buffer and its exact layout for
   both the fused `Q8_1Block` path and the MMQ `MxQ8_1MmqBlock` path.
3. MMQ Gate and Up projection outputs and post-SwiGLU activation.
4. Fused Gate/Up/SwiGLU output.
5. The layer output and recurrent state boundary needed to confirm propagation.

The first route-specific kernel is only classified after exact input/history
identity and capture provenance are verified. Compare each route with a CPU
reference that consumes the captured production quantized activations and
canonical GGUF weights. Do not compare outputs computed from different inputs.

## QKV references

For the same captured layer-0 position-0 input and Q6_K weights, retain:

- **A — Full-precision input:** canonical dequantized Q6_K dot original F32
  activation, accumulated in FP64 and rounded to F32.
- **B — Quantization-aware input:** the same weights dotted with the exact
  dequantized activation represented by the production MMQ Q8 buffer,
  reproducing its scales, rounding, block layout, and accumulation contract.
- **C — Production GPU:** the existing Q6_K Mx MMQ result.

Report `||B-A||2/||A||2` as activation quantization contribution and
`||C-B||2/||B||2` as kernel contribution. Verify B against independent test
vectors and the captured quantizer bytes. Do not transfer M4-A tolerances or
classify the result using an unvalidated approximate Q8 quantizer.

## Gates and disposition

- **Identity gate:** exact model/prompt/history and compatible run identities;
  same forced token history through the captured position.
- **Capture gate:** all expected files have declared shape, dtype, byte count,
  and SHA-256; no non-finite values; no overwritten prior evidence.
- **Reference gate:** CPU reference consumes the exact production activation
  bytes and canonical weights, with assert-based synthetic layout checks.
- **Safety gate:** correct BDF and pinned image; idle/thermal guard passes;
  all guarded process groups exit cleanly.
- **Stop:** first causally attributable defect; an independently justified
  numerical classification with route parity still blocked; or `UNRESOLVED`
  with the exact missing evidence and a documented stopping decision.

Exact full-model token parity remains required. No tolerance widening,
performance claim, or production selection change is authorized by this task.
