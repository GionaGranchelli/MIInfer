# M31-0002T-0001 — Numerical and state correctness audit

Status: no confirmed runtime defect; sampler configuration mismatch confirmed;
device logits remain unmeasured. Read-only audit; no GPU workload or source
change was made.

## Measured execution path

The checkpointed harness creates a fresh `PrefillV2Model`, resets state,
disables prefix reuse by default, and enables HIP-graph decode. A 1,024-token
prompt is processed in two eager 512-token prefill chunks. MIInfer computes the
first output from final-prompt logits; subsequent outputs are selected after
one-token graph replays. Prompt positions are 0–1023; generated-token decoding
starts at position 1024. The source progression and graph attention's
inclusive `position + 1` bound are consistent with causal decode.

The run's harness does not override the default repetition penalty (1.15),
while llama.cpp uses penalty-free greedy sampling. This is the confirmed
configuration mismatch; it is plausibly causal at step 7 because llama's
candidate 248046 was emitted at step 2, but the raw candidate logits are not
available. Do not infer an execution-path error from the selected token alone.

MIInfer loads mixed Q4_K_M GGUF tensors and repacks them for its MMQ and fused
Wave consumers. Its recurrent prefill and recurrent decode use different
kernels, so if raw-logit disagreement remains after sampler alignment, the
prefill-to-decode transition and recurrent update are appropriate next
comparison points. Source inspection alone does not show a broken transition.

## Recent M31 fixes and applicability

- M31-0004 fixes reusable-prefix residency invalidation. The measured call is a
  fresh cold call with reset enabled and no saved reusable prefix, so no
  specific causal path from that bookkeeping fix to this divergence is
  established. Its GPU correctness cases remain unvalidated.
- M31-0005 fixes the eager suffix attention caller's inclusive KV boundary.
  The compared call uses graph decode, whose dynamic attention path already
  uses `position + 1`; that eager-only correction does not explain this run.
  Its poisoned-future-slot GPU regression remains unvalidated.
- No evidence in the reviewed path confirms a GDN, RoPE, KV ownership, graph
  replay, or quantized-matmul runtime defect. These remain follow-up areas only
  if aligned raw logits show a material disagreement.

## Required correctness evidence

First remove the sampler mismatch. Then compare both engines on identical
forced token prefixes and capture logits before sampling. If differences are
material, find the earliest practical boundary with a targeted comparison:
post-prefill hidden/logits, then recurrent state/KV boundary, then selected
layer outputs. Avoid dumping every activation. The test should cover first
token-after-prefill and the step-7 prefix, and report exact positions and
top-candidate margins.

The current MIInfer binary hash is recorded, but the reduced report does not
bind that binary to a full source-tree SHA. Source conclusions here apply to
the inspected working tree; that provenance gap must be closed for a
reproducible follow-up.
