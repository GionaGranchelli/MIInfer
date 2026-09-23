# EXP-0378 — Determine Whether a Non-Scalar B64 Core Contract Exists

## Question

Does MIInfer already contain the mathematical/runtime primitives needed for a
complete 64-token model chunk, with only higher-level guards preventing their
composition?

## EXP-0377 rejection basis

EXP-0377 scheduled an exact P960 B64 chunk, but the runtime reported:

```text
partial_b64=1
scalar_layer_run_calls=64
scalar_layer_run_tokens=4096
```

The scheduler label was correct, but every layer still used scalar
`layer.run()`. EXP-0378 audits the underlying primitives without changing any
production scheduler or routing behavior.

## Method and gates

- Source audit of recurrent and attention guards, launch wrappers, grids, and
  workspace sizing.
- Existing GPU primitive test temporarily exercised at `kTokens=64` and then
  was restored byte-for-byte. The test-only run was not a production change.
- Existing GDN chunk oracle already exercises `chunk=64`.
- No new kernel, persistent buffer, scheduler change, or default behavior.

## B64 capability matrix

Legend:

- **A — B64_EXISTING_AND_PROVEN:** existing primitive executed at 64 with
  correctness checks.
- **B — B64_BLOCKED_BY_POLICY_GUARD:** underlying primitive accepts the shape,
  but the current composition guard requires 128 or the existing test did not
  exercise the full model-family combination at 64.
- **C — B64_PRIMITIVE_MISSING:** no such primitive was found. None was found.

| Layer family | Stage | Category | Evidence |
| --- | --- | --- | --- |
| Recurrent | input/RMS norm | A | `launch_qwen3_rms_norm_batch`; count-64 temporary GPU test passed |
| Recurrent | QKV/Z projection | B | existing MMQ batch wrappers accept 1..512 and mask the second 64-row half; production wide guard is `token_count >=128`; count-64 MMQ test passed |
| Recurrent | beta/decay | B | dual batched GEMM and `prepare_beta_decay` are token-count parameterized; `prefill_wide_beta_decay` requires `>=128` |
| Recurrent | convolution/history | A | count-64 temporary convolution batch test passed; kernel loops token_count and updates absolute history positions |
| Recurrent | GDN/state update | A | `launch_m12_gdn_chunk` requires exactly `chunk_size=64`; GDN oracle passed with chunk 64 and finite state/output |
| Recurrent | SSM output projection | B | existing Q5/MMQ and batched GEMM wrappers accept count 64; `prefill_wide_ssm_out` requires `>=128`; count-64 MMQ test passed |
| Recurrent | residual/post norm | A | existing fused batch norm/add accepts arbitrary positive token_count; count-64 GPU test passed |
| Recurrent | FFN gate/up | B | existing MMQ batch primitives accept count 64; production wide composition is guarded by the wide path minimum |
| Recurrent | SwiGLU | A | existing elementwise path is count/element based; count-64 deferred tail test passed |
| Recurrent | FFN down | B | existing Q4/Q6 MMQ batch primitives accept count 64; no B64 whole-layer composition currently selects them |
| Recurrent | final output | A | existing batch residual/output primitives are count based; no B64-specific invariant found |
| Attention | Q/K/V/gate projection | B | count-64 preparation path exists and count-64 MMQ primitives passed; wide preparation branch is selected only at `>=128` |
| Attention | Q/K norm and RoPE | B | batch wrappers accept arbitrary positive token_count; the wide composition guard prevents selection at 64 |
| Attention | KV write | B | batch KV-store wrappers validate capacity/base, not a 128 minimum; composition is blocked by the wide preparation guard |
| Attention | causal attention | A | exact 64-token causal fixture at non-zero base passed against scalar reference |
| Attention | O projection | B | existing batched MMQ/GEMM primitives accept count 64; wide whole-layer composition is not selected at 64 |
| Attention | FFN/output | B | existing count-based batch/MMQ tails accept 64; upstream scalar `layer.run` prevents the complete route |

No category C primitive was identified. The categories marked B are
composition/policy blockers or lack a full model-family count-64 fixture, not
mathematical impossibilities.

## Recurrent audit details

The recurrent wide entry points all contain the same policy boundary:

```text
prefill_wide_qkv_gate: token_count < 128 -> reject
prefill_wide_beta_decay: token_count < 128 -> reject
prefill_wide_causal: token_count < 128 -> reject
prefill_wide_ssm_out: token_count < 128 -> reject
prefill_wide: token_count < 128 -> reject
```

Their underlying operations are token-count parameterized. The Q4/Q5/Q6 MMQ
wrappers accept `token_count <= 512`; the MMQ kernel launches a 128-token tile
but explicitly guards both `token_base` and `token_base + 64`, so a 64-token
call writes only the valid first half. The recurrent causal wrapper uses the
existing 64-token chunk primitive. Its workspace is explicitly documented as
one 64-token chunk per value head.

The existing count-64 deferred-tail APIs therefore do not imply a complete
B64 layer: they cover the tail after scalar `layer.run`, while the wide
projection/core composition remains guarded at 128.

## Attention audit details

Attention preparation accepts `count == kPrefillBatch` (64) or the configured
capacity. Its low-level batch wrappers validate positive token count and cache
bounds rather than requiring 128. The causal batch kernel uses one block per
token/query-head pair and was directly tested with 64 queries at a non-zero
base and causal cache bounds.

The current `batched_attention` selector adds `count >= kM12PrefillBatch`
(128), so these existing count-64 primitives are not composed by the normal
wide route. This is a policy/composition guard, not a discovered kernel
invariant.

## Focused probe results

Temporary count-64 execution of the existing
`miinfer-qwen35-conv-batch-test` passed. It covered existing convolution,
Q/K normalization, dual projection, RMS/add-norm, Q8/MMQ Q4/Q5/Q6 paths,
fused Q/K split normalization/RoPE/KV storage, and causal attention. The
attention portion was separately exercised with 64 queries, base 3, and cache
capacity 80; it passed its scalar reference checks. The source test was
restored to its original constants afterward.

The existing GDN oracle passed independently:

```text
tokens=128, chunk=64
max_output_error=1.21072e-08
max_output_relative_error=7.567e-06
max_state_error=1.3411e-07
```

The GDN chunk benchmark also reports the fixed 64-token chunk path and finite
output/state errors. These are primitive existence/correctness results, not
performance promotion.

## Workspace/resource audit

Existing recurrent and attention workspaces are sized to the configured
prefill capacity and therefore contain at least 64 rows. The GDN workspace is
explicitly 64-token. No B128-only stale-slot dependency was found in the
count guards or kernel output bounds. No new allocation or persistent buffer
was added. No new launch geometry was introduced, so no new compiler resource
qualification was required; the audited kernels retain their existing
compiled resources.

## Decision

**QUALIFY_CONTRACT_EXISTS.** The repository contains the required underlying
count-64 primitives. EXP-0377 failed because the complete B64 composition is
blocked by higher-level `>=128` guards and the layer-major caller falls back
to scalar `layer.run`; it did not demonstrate a missing GPU math primitive.

## Exact next PRIMARY

EXP-0379 — compose the existing B64 primitives by minimally relaxing the
proven policy/dispatch guards, then qualify complete P960/P1022 semantics and
performance. Keep B4 scheduling out of scope.

No production scheduler behavior or new GPU math kernel was implemented.
