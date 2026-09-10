# EXP-0286 — M23 wide-prefill operation mapping

**Status:** RUNNING  
**Milestone:** M23  
**Date:** 2026-09-09  
**Baseline commit:** `ebeb0ffa1b8a08683bf05888fc6992ddf661a3c7`

## Question

Can one recurrent Qwen3.5 layer consume a B128 prompt chunk without
traversing its projection weights once per B4 group?

## Source

`mx-llama.cpp` at `2e9d29fe736969160f17476ec6f0a6298cee6966`:

* `src/models/qwen35.cpp:340-467` builds every token-local projection with
  `build_lora_mm(weight, cur)`, where `cur` contains the complete ubatch.
* Only `build_conv_state`/`ggml_ssm_conv` and `build_recurrent_attn` retain
  causal state.  Their inputs have already been projected for the full chunk.
* `build_layer_ffn` invokes parallel Gate/Up and then Down over that same
  matrix.

## Required operation mapping

| mx operation | MIInfer current path | required M23 replacement |
|---|---|---|
| `wqkv = MM(wqkv, X[B])` | `RecurrentLayer::prepare_prefill_batch`; native Q4/Q6 B4 GEMV loop | one B128 matrix projection into `prefill_qkv` |
| `z = MM(wqkv_gate, X[B])` | same B4 loop / combined QKV+gate layout | one B128 matrix projection into `prefill_gate` |
| `beta = MM(ssm_beta, X[B])` | per-token `launch_qwen35_f32_dual_gemv` | one B128 beta matrix projection, then prepare beta |
| `alpha = MM(ssm_alpha, X[B])` | per-token `launch_qwen35_f32_dual_gemv` | one B128 alpha matrix projection, then prepare decay |
| causal convolution | `RecurrentLayer::run` once per token | one chunk convolution preserving the four-token history |
| GDN scan | staged per token then B4 core, or M12 chunk scan | M12 chunk scan over the already-projected B128 tensors |
| `ssm_out = MM(ssm_out, output[B])` | `finish_prefill_batch4` Q5 B4 GEMV | one B128 matrix projection |
| `ffn_gate = MM(ffn_gate, X[B])` | `finish_prefill_batch4` Q4 B4 GEMV | one B128 matrix projection |
| `ffn_up = MM(ffn_up, X[B])` | `finish_prefill_batch4` Q4 B4 GEMV | one B128 matrix projection |
| `ffn_down = MM(ffn_down, SwiGLU[B])` | `finish_prefill_batch4` Q4/Q6 B4 GEMV | one B128 matrix projection |
| attention `Q+gate+K = MM(wqk, X[B])` | `FullAttentionLayer::prepare_prefill_batch`; Q/K B4 GEMV loop | opt-in B128 Q4 MMQ projection |
| attention `V = MM(wv, X[B])` | same B4 GEMV loop | opt-in B128 Q4/Q6 MMQ projection |

## Dispatch contract

For a B128 recurrent layer, each listed weight-bearing family must dispatch
once (or a documented small tile count), never 32 B4 groups.  The M23 opt-in
is `MIINFER_PREFILL_WIDE_CHUNK=1`; decode remains on the existing path.
Set `MIINFER_PREFILL_TRACE_DISPATCH=1` to print the per-chunk projection
dispatch counters while profiling.
Wide attention emits a matching `M23 attention dispatches` line covering QK,
V, Q/K postprocess, causal attention, O, and FFN Gate/Up/Down launches.

## Baseline

M22's best B4 architecture measured P512 at 55.72 tok/s.  The first gate is
one B128 recurrent layer at least 3x faster than its B4 equivalent while
matching output, convolution state, and GDN state.

## Decision

Proceed with a one-layer, FP16-staged wide prototype.  It may use bounded
workspace, but must not add a permanent FP16 model copy or change decode.

## B128 convolution contract

`miinfer-qwen35-conv-batch-test` compares one `B128` wide convolution launch
against 128 canonical single-token launches using the same raw QKV sequence,
weights, and four-slot history.  It checks every split Q/K/V value and the
final circular history.  This validates the causal part of the M23 schedule
without using B4 as its execution unit.

The same test also compares `launch_qwen35_f32_dual_gemm_batch` with 128
canonical dual-GEMV calls.  The wide operation emits beta and alpha raw values
with one dispatch for the pair, before the existing elementwise beta/decay
preparation.  This is the M23 replacement for the prior per-token beta/alpha
launches; it is deliberately separate from the large quantized projections.

`launch_qwen3_rms_norm_batch` similarly matches 128 canonical RMSNorm calls.
`RecurrentLayer::prefill_wide_beta_decay` now composes the three operations:
wide RMSNorm, one dual beta/alpha projection dispatch, and elementwise
beta/decay preparation.  Its buffers are the same B128 causal-core buffers;
no decode buffer or model-weight copy was added.

## mx gfx906 MMQ contract

The pinned source is specific enough to prevent an M13 repeat:

* `ggml-cuda/q8_repack/mul-mat.cu` takes the tiled path when `B > 8`.
* At B128 it quantizes the full activation matrix to grouped `Q8_1` once,
  then dispatches one Q4_K/Q5_K/Q6_K GEMM per weight tensor.
* `repack-common.cuh` fixes the relevant geometry to `BK=4`, `BM=64`,
  `TN=2`, and `NROW_LANES=4`; the B128 launch therefore covers a 64-output
  by 128-token tile with a `dim3(64, 4)` Wave64 workgroup.
* mx's resident repacked weights serve both prompt and decode paths.  M23's
  temporary prototype may stage one layer, but it must not copy a tensor per
  prompt chunk or retain a second full model.

MIInfer's next large-projection primitive must use this grouped-activation,
64-row tiled shape (starting with the actual Q6_K/Q4_K recurrent inventory),
not the rejected canonical-layout B64/B512 mapping from M13.

`M23Q8_1MmqBlock` and `launch_m23_q8_1_mmq_quantize` now materialize the
required B128 grouped activation input: four 32-value Q8_1 groups, each with
its scale and sum, per 128-value block.  The B128 test verifies every scale,
sum, and quantized value against an independent host calculation.  This is
bounded activation workspace, not a resident weight copy.

For the permitted first-layer staging proof, `MIINFER_PREFILL_WIDE_CHUNK=1`
now expands only recurrent layer 0's QKV and Z tensors once at model load and
keeps them as FP16 (`~168 MiB`). `prefill_wide_qkv_gate` invokes each staged
matrix once over B128 through rocBLAS, sharing the existing bounded input
workspace and the wide beta/alpha preparation. It does not change decode or
stage any other layer. The complete tail is now connected through
`RecurrentLayer::prefill_wide`: wide SSM-out, batched residual/post-attention
RMSNorm, wide FFN Gate/Up, SwiGLU, and wide FFN Down. The initial prototype
dispatched that method only for recurrent layer 0 and only for complete B128
chunks; later layers and decode then continued through their existing paths.

The same layer-0 prototype now stages Q5_K `ssm_out` once (an additional
~60 MiB) through the MIInfer-owned Q5_K→FP16 converter and provides a single
B128 `prefill_wide_ssm_out` GEMM. The complete layer also stages its FFN Gate,
Up, and Down tensors once. The selected layer's persistent FP16 staging is
approximately 762 MiB, bounded to one layer; no prompt-time weight copy and no
whole-model FP16 copy exists.

## Integration result — 2026-09-09

The real target GGUF loads with the opt-in below and allocates
28,126,601,556 bytes (26.20 GiB). The M23 switch also enables the established
B128 M12 scratch/scan configuration for the unchanged later layers, avoiding
a 64-token scratch-capacity page fault:

```text
MIINFER_PREFILL_LAYER_MAJOR=1
MIINFER_PREFILL_WIDE_CHUNK=1
MIINFER_PREFILL_CHUNK=128
```

An exact B128 PP-only request now completes at 2,672.89 ms / 47.89 tok/s.
This is an execution smoke result, not a performance result: a one-token
continuation throws `token ID is outside tokenizer vocabulary`. The wide layer
therefore fails the required qualified-continuation gate.

`MIINFER_WIDE_VALIDATE=1` compares the wide layer against 128 canonical layer
calls from the same saved state/history. The measured errors are
`qkv_max_abs=59.249`, `output_max_abs=7708.48`, `state_max_abs=26.7651`, and
`history_max_abs=57.0098`. The first material divergence is therefore the
FP16 staged QKV projection, not the causal scan. Raw-FP16 staging cannot be
the correctness route; M23 must move to a Q8-activation K-quant MMQ reader.

Decision: RETEST. The next gate is a layer-0 output/state handoff check against
the existing recurrent oracle before any performance comparison or rollout.

2026-09-09 update: added opt-in `MIINFER_PREFILL_WIDE_MMQ_QKV=1` for layer-0
Q4_K QKV/Z. It consumes the grouped M23 Q8_1 pack and has the pinned mx
64-row × 128-token / 64×4-thread geometry. The focused GPU test compares its
synthetic B128 Q4_K output against the canonical Q8_1 MMVQ reader. This is a
correctness foothold only; it is not profiled or accepted, and the Q5_K
SSM-out/FFN FP16 stages still prevent a qualified full-layer result.

The first exact-B128 Q6_K validation then reduced QKV/Z error to `0.002594` /
`0.001726`, but exposed a separate wide beta/alpha preparation bug:
token-major indexing read the 48-element `dt` and `a` vectors with a flattened
B128 index, producing `decay_max_abs=0.99982`. The kernel now maps that index
back to the static 48-head dimension. This is a correctness repair, not a
performance result.

After that repair, the exact P128 validation reports `qkv_max_abs=0.002594`,
`gate_max_abs=0.001726`, `beta_max_abs=1.19e-7`, `decay_max_abs=1.79e-7`,
`state_max_abs=0.001300`, and `history_max_abs=3.43e-5`. Thus the quantized
QKV/Z and causal core now have a qualified numerical handoff candidate. The
remaining `output_max_abs=10647.4` is downstream in the still-FP16 Q5_K
SSM-out and FFN tail, so this does not satisfy the complete-layer or
continuation gates.

`MIINFER_PREFILL_WIDE_MMQ_SSM_OUT=1` now independently enables the matching
Q5_K SSM-out reader. It is build-tested but awaits the same B128 canonical
reader comparison before its result is accepted.

The exact P128 Q5_K run completed with continuation `z` and
`output_max_abs=5.54105`; it reduced the prior FP16-tail output error by four
orders of magnitude while preserving the QKV/Z/state figures above. The next
isolated opt-in is `MIINFER_PREFILL_WIDE_MMQ_FFN=1`, which uses Q4_K readers
for Gate/Up and the existing Q4_K/Q6_K reader for Down.

With all three MMQ opt-ins enabled, the complete layer-0 B128 validation
reports `output_max_abs=0.0090065`, `state_max_abs=0.00130033`, and
`history_max_abs=3.43e-5`; the generated continuation is `z`. This completes
the numerical/dataflow proof for one recurrent layer. Its P128 end-to-end
smoke is only 41.16 tok/s, however, so the direct canonical-layout readers are
REJECTED for the >=3x layer-speed gate. Keep them as correctness references;
the next candidate must add the pinned reference's load-time row/sub-block
repack and LDS weight-tile reuse rather than tuning this mapping.

## Re-evaluation — 2026-09-10

The row/sub-block repack was extended to compressed MIInfer-owned Q4/Q5/Q6
MMQ tiles and widened to every recurrent layer. An allocation branch for the
combined QKV/Z layout initially skipped host packing of Z; an exact B128
request then failed at the Q4 wrapper with a null weight pointer. Packing now
runs after the combined/non-combined branch, and the exact 128-token request
completes across all layers.

The all-layer scratch run used `MIINFER_CONTEXT_CAPACITY=1024`, the three M23
MMQ switches, and `MIINFER_PREFILL_WIDE_REPACKED_MMQ=1`. It allocated
23,011,004,756 bytes and completed P128 in 6,029.50 ms (21.23 tok/s). The
profiled run measured 5,181.24 ms wall time; recurrent projection GPU time
was dominated by FFN Gate/Up (898.8 ms summed) and Down (586.0 ms), while
allocation/upload/free overhead remained substantial.

Layer-0 validation with the full all-layer path still reports
`qkv_max_abs=7.91361`, `output_max_abs=0.929607`, `state_max_abs=5.27815`,
and `history_max_abs=7.13091`. The same QKV error remains when the new
repacked reader is explicitly disabled, so it is attributable to the existing
wide Q8/MMQ-vs-token projection path rather than the null-pointer repair.
This candidate is therefore **REJECTED** for correctness and the >=223 tok/s
qualification. Keep the compressed-tile host/kernel tests and constructor
repair; do not claim a performance win. A future candidate must first make
the wide Q8 projection numerically match the recurrent oracle and eliminate
per-layer scratch churn before a P512 comparison.

## Re-evaluation — full layer-major scheduling — 2026-09-10

The temporary repacked buffers now use a process-local shared pool, removing
the synchronous `hipMalloc`/`hipFree` pair for every layer/chunk. The larger
structural change is opt-in `MIINFER_PREFILL_FULL_LAYER_MAJOR=1`: embeddings
are materialized once, then each layer consumes all four B128 chunks before
advancing to the next layer. Repacked weights therefore remain resident for
the complete P512 traversal instead of being uploaded four times.

On the exact 512-token prompt, this reduced wall time from 18,839.31 ms
(27.18 tok/s) to 14,357.14 ms (35.66 tok/s), with 23,026,733,396 bytes
allocated. The P128 layer-0 validation through the reordered scheduler
preserved the prior figures (`qkv_max_abs=7.91361`,
`output_max_abs=0.929607`, `state_max_abs=5.27815`,
`history_max_abs=7.13091`). Loop reordering is numerically stable, but the
wide Q8/MMQ projection mismatch remains and 35.66 tok/s is still far below
the 223 tok/s target. KEEP this scheduler as the optimization base; the next
required experiment is a B256/B512 projection kernel that reuses a staged
weight tile across more than 128 tokens.

The same exact P512 prompt with `--max-tokens 1` completed the continuation
check (token text `The`) in 13,690.75 ms prefill plus 1.51 ms decode. This
confirms the reordered path reaches the existing logits/decode handoff; it is
not evidence of numerical parity with the token-major oracle.

### Rejected micro-optimization — activation-sum hoist

The affine repacked kernel was briefly changed to compute each Q8 group sum
once in shared memory instead of once per output row. Exact P128 regressed
from 27.28 tok/s to 22.33 tok/s, so the added shared pass and synchronization
outweighed the saved integer dots. The change was reverted; **REJECT** for
this tile shape.

### Rejected experiment — paired 128-row MMQ workgroup

The repacked kernels were temporarily changed to a 512-thread `(64,8)`
workgroup containing two 64-row weight tiles. An exact P128 smoke appeared
faster (28.40 tok/s), but the focused Q4 MMQ test found a zeroed token-64
output. The mapping was reverted; larger workgroups remain rejected until the
shared activation/output indexing is corrected.

### Rejected experiment — cached affine activation sums

The affine MMQ path was changed to cache each Q8 group sum once, first in the
activation block and then in a standalone post-quantization kernel. Packing
correctness remained valid, but same-prompt P512 measurements (33.29, 36.20,
35.14 tok/s for the in-block variant; 34.70 tok/s for the standalone pass)
did not beat the 37.33 tok/s control. Both variants were reverted.

### Rejected experiment — B256 staged MMQ tile

The repacked Q4/Q5/Q6 MMQ kernels were widened from two 64-token output
columns to four, staging 256 activations while retaining one weight tile.
An exact 512-token smoke run measured 21,087.43 ms (24.28 tok/s), versus the
35.66 tok/s full-layer B128 scheduler baseline. The widened kernel was
reverted; the larger per-thread accumulator/LDS footprint is not a win on
gfx906 in this workload.

### Candidate — mx-style 128-row Q4/Q5 MMQ mapping — 2026-09-10

Q4_K and Q5_K were remapped to a 256-thread `(64,4)` workgroup. Each block
loads two adjacent 64-row weight tiles, expands them once in LDS, and reuses
the tile across four warps of token outputs. The focused Q4/Q5 correctness
test passed.

An initial exact P512 prompt measured **13,410.92 ms / 38.18 tok/s**, compared
with the same-prompt full-layer B128 control at **37.33 tok/s**, but that run
is invalidated by the later LDS bounds finding below. **RETEST** before any
performance conclusion; full validation remains outside the qualified path
(`qkv_max_abs=7.91361` in the existing wide oracle check).

### Candidate — Q6 row-128 MMQ mapping — 2026-09-10

The Q6_K repacked reader now uses the same 256-thread `(64,4)` geometry as
the Q4/Q5 candidate: two 64-row weight tiles are expanded once and reused
across four token-output warps. The legacy Q6 reader remains in the binary as
an unselected fallback until a GPU run proves numerical parity. This change
builds successfully but has no benchmark or correctness result while `/dev/kfd`
is unavailable. **RETEST** with the exact command above.

### Candidate — B128 attention Q/K/V projection — 2026-09-10

The wide recurrent path was not the only remaining projection loop: full
attention layers still ran Q/K/V GEMVs in B4 while `run()` consumed each token.
With `MIINFER_PREFILL_WIDE_ATTN=1`, Q+K and V are packed into the existing
M23 MMQ layout and projected once per B128 chunk; causal attention and KV
writes remain position-ordered. The path builds, but the GPU became
unavailable after the cold benchmark launches, so no performance or numerical
parity result is claimed yet. **RETEST** when a valid MI50 session is
available. The scheduler also now passes the prepared normalization buffer to
`run()`, avoiding a second per-token RMSNorm launch after any batched prepare.
The same opt-in now batches the attention O projection and FFN Gate/Up/Down
tail through MMQ after the batch causal-attention launch; this is
build-verified but still awaits a valid GPU session for correctness and speed
measurements.
The ordered attention loop is now eliminated after projection: two batch
postprocess launches normalize/split/RoPE Q+gate and K, write the KV cache, and
populate the query buffers; one causal Wave64-per-query-head launch then scans
each token's KV prefix. This removes decode attention and Q/K postprocess
dispatches from the B128 loop; it is also build-verified only.

Retest command (exact P512 prompt):

```sh
MIINFER_CONTEXT_CAPACITY=1024 \
MIINFER_PREFILL_LAYER_MAJOR=1 MIINFER_PREFILL_WIDE_CHUNK=1 \
MIINFER_PREFILL_FULL_LAYER_MAJOR=1 MIINFER_PREFILL_WIDE_ATTN=1 \
MIINFER_PREFILL_WIDE_REPACKED_MMQ=1 \
MIINFER_PREFILL_WIDE_MMQ_QKV=1 MIINFER_PREFILL_WIDE_MMQ_SSM_OUT=1 \
MIINFER_PREFILL_WIDE_MMQ_FFN=1 \
miinfer run Qwen3.8-27B-Q4_K_M.gguf --prompt "$(seq 1 51 | awk \
'{printf "The quick brown fox jumps over the lazy dog. "}')The quick" \
--max-tokens 0
```

### Re-evaluation — row-128 LDS bounds repair — 2026-09-10

The initial row-128 smoke result (38.18 tok/s) is **invalidated**: each
64-token workgroup tile loaded `token_count` activations, overrunning the
64-entry shared activation array on the first half of a B128 launch. Both Q4/Q5
and Q6 readers now cap the load at 64 tokens. The prior timing must be rerun;
no performance claim survives this correction.

### Candidate — batched causal attention launch — 2026-09-10

The full-attention M23 path now runs Q+gate normalization/RoPE, K
normalization/RoPE plus KV stores, and causal attention as batch launches over
the prepared B128 projections. A single Wave64-per-token/query-head kernel
scans each token's causal KV prefix and writes the gated attention matrix
consumed by the existing batched O/FFN tail.
The decode attention path is unchanged. Build and host tests pass; GPU
correctness and timing remain **RETEST** items because this session exposes no
ROCm-capable agent to HIP.

The GPU batch test now includes a nonzero-base-position B8 causal-attention
case against a CPU softmax reference; it will exercise the new cache-prefix
and token-stride contract for both FP32 and the default FP16 KV cache when a
device is available. The same test now checks batched Q/K normalization,
absolute-position RoPE, KV writes, and the corrected full-row `qfull` stride
against host calculations, including the FP16 KV-store conversion path.
Its synthetic repacked Q4_K/Q6_K MMQ case uses 128 output rows, exercising both
64-row halves of the production row-128 mapping.
The canonical Q6 reference differed by only about `2.4e-6` at one widened-row
accumulation (different reduction order), so MMQ output comparison uses a
`1e-5` relative/absolute floor while the non-MMQ checks retain their tighter
tolerance.

When a gfx906 agent was briefly exposed on 2026-09-10, the focused test and
`qwen35-conv-batch-gpu` CTest both passed. The agent then disappeared before
the model-sized P512 launch could allocate its full working set, so this is
kernel-level GPU correctness evidence only; end-to-end timing remains
**RETEST**.

### Re-evaluation — batched Q/K projection stride repair — 2026-09-10

Source audit found that the first batch Q/K postprocess draft advanced the
per-token `qfull` pointer by only the Q+gate rows, omitting the K tail. That
would have corrupted every token after the first. The stride now includes
`(query_heads * 2 + kv_heads) * head_dim`; the fix was applied before any GPU
timing or correctness result, so no invalid measurement is retained.

### Candidate — mx BM64×BN128 direct-global MMQ — 2026-09-10

The active Q4/Q5/Q6 wrappers now use one 64-row weight tile for 128 tokens,
matching mx's BM64×BN128 geometry so each weight tile is read once per B128
projection. Threads expand coalesced global tile loads directly into the
packed LDS planes, reducing the shared footprint from about 35–40 KiB to
26 KiB and avoiding an occupancy collapse on gfx906. The 128-row focused GPU
test passes after this change; model-sized timing remains **RETEST**.

### Candidate — Wave64 MMQ metadata broadcast — 2026-09-10

The active BM64×BN128 kernels previously reloaded identical per-row scale
metadata in every lane. Lane 0 now performs each metadata load and broadcasts
it across the Wave64, reducing redundant global transactions and scalar load
instructions without changing the accumulation mapping. The release build and
all 11 host-only tests pass; the focused GPU test and P512 A/B timing remain
**RETEST** because HIP allocation is currently unavailable.
The focused `qwen35-conv-batch-gpu` CTest also passed once after the change;
the subsequent exact-model attempts lost the ROCm agent during startup.

### Candidate — MMQ activation-scale hoist — 2026-09-10

The active kernels now round each staged activation scale to FP16 once per
token/slab/K tile and reuse the resulting float across all output rows. This
removes invariant LDS loads and conversions from the inner row loop while
preserving the canonical half-rounded scale. Release build, host-only CTest,
and `qwen35-conv-batch-gpu` pass; exact P512 timing remains **RETEST**.

### Re-evaluation — non-B128 attention dispatch guard — 2026-09-10

The wide attention tail was being selected for 64-token chunks even though
wide QK/V preparation is currently implemented only for B128. That mixed the
per-token QK/V path with the wide post-attention tail and made dispatch traces
misleading. Both layer-major schedulers now require `count == 128` before
selecting the wide attention tail; smaller chunks use the existing ordered
attention path and batched O/FFN tail. The release build and all 11 host-only
tests pass. GPU tests are **RETEST**: this run reports no ROCm-capable device.
The attention dispatch trace also now allocates nine counters (including FFN
Down); the previous eight-entry array was an out-of-bounds write that could
corrupt the trace and adjacent state.

### Re-evaluation — Q6 MMQ high-nibble parity and attention scratch lifetime — 2026-09-10

The varied Q6 fixture exposed a real repacked-kernel error: the LDS unpacker
read the low nibble for both 16-value halves, while the packer stores the second
half in the high nibble. The Q6 unpackers now select `4 * half`; the focused
GPU test passes with varied payloads and canonical parity. Attention MMQ tiles
are now host-packed and uploaded through a six-slot layer-scoped scratch pool,
then released after each layer, avoiding a second persistent copy of every
attention projection and allowing the model to fit its context-512 working set.
End-to-end P512 timing is still **RETEST**.

The model-sized smoke was also rerun at context capacity 512 with `hello`:
initialization completed with `device_allocated_bytes=23138404692` and one
prefill token, confirming the scratch-pool lifetime fits. The exact 512-token
prompt could not be timed because the ROCm agent disappeared before that
launch; no parity or throughput claim is made.

### Re-evaluation — exact P512 timing and row-128 A/B — 2026-09-10

The exact Qwen3.8-27B-Q4_K_M P512 prompt was rerun on the exposed gfx906
device with all M23 wide/repacked families enabled. The corrected row-64
BM64xBN128 mapping measured `15,896.04 ms / 32.21 tok/s`; a separate profile
run measured `14,297.71 ms / 35.81 tok/s` while sampling one recurrent layer.
Dispatch tracing showed one qkv, gate, beta/alpha, ssm-out, FFN gate, FFN up,
and FFN down launch per recurrent B128 chunk, plus one launch for each
corresponding full-attention family. The model allocated
`23,138,404,692` device bytes at context capacity 512.

The opt-in row-128 variant (`MIINFER_M23_REPACKED_ROW128=1`) passed the focused
varied-Q6/Q4 GPU parity test and measured `14,092.29 ms / 36.33 tok/s` on the
same prompt. Its profiled FFN gate/up and down shares were not lower than the
row-64 run, so the option remains disabled by default and is **RETEST/REJECT**
as a parity solution. Both candidates remain far below the pinned mx-llama
`222.64 tok/s` target. The sampled profile identifies recurrent FFN gate/up
(`44.1%`) and FFN down (`19.4%`) as the dominant next optimization target.

### Rejected experiment — packed-Q4 LDS experiment — 2026-09-10

The dominant recurrent FFN Gate/Up family is Q4_K. An opt-in
`MIINFER_M23_REPACKED_PACKED_Q4=1` kernel now stages resident Q4 nibble words
directly in LDS and masks them at `sdot4` time, instead of expanding each tile
into an 8 KiB byte plane. This is the minimum implementation change that
follows mx's packed `dp4a` data path while preserving the validated expanded
kernel as the default and retaining a one-variable A/B switch. The release
build and host tests pass; this section records the isolated A/B below.

### Re-evaluation — packed-Q4 scope repair and P512 A/B — 2026-09-10

The first packed-Q4 A/B was invalid as a whole-path comparison: the switch
also selected packed Q4 for QKV, and profiling showed QKV rising from
`3.97 ms` to `23.14 ms` in the sampled layer. The dispatch gate now limits the
experiment to the exact Qwen3.8-27B FFN Gate/Up (`17408x5120`) and Down
(`5120x17408`) shapes; QKV and attention use the validated expanded reader.
The focused GPU test passes with the packed switch enabled.

On the exact P512 prompt, corrected row-64 controls measured `32.21` and
`32.94 tok/s`; packed-FFN candidates measured `34.32` and `35.04 tok/s`, but
the sampled FFN Gate/Up kernel was slower (`20.38 ms` versus `16.91 ms`). The
whole-run spread is therefore run-state noise, not a repeatable kernel win.
Allocation stayed at `23,138,404,692` bytes. The packed-Q4 experiment is
**REJECTED** and its code was removed; the validated expanded reader remains
the only production path. The next gap remains the full MMQ mapping, not
another global LDS micro-optimization.

### Candidate — gfx906 affine MMQ launch bounds — 2026-09-10

The active repacked affine MMQ kernel now declares `__launch_bounds__(256, 1)`
to match the pinned gfx906 MMQ configuration and give LLVM an explicit VGPR /
occupancy contract. The focused varied-payload GPU test passes. On the exact
P512 prompt, the same row-64 path measured `43.17` and `39.68 tok/s` with the
attribute, versus corrected controls of `32.21` and `32.94 tok/s`; pair means
improve by approximately `27.2%` with no VRAM change. **KEEP** as the default
repacked affine MMQ mapping. The result is still only `18.6%` of the pinned
`222.64 tok/s` reference, so it is an intermediate optimization rather than
parity.

### Candidate — gfx906 Q6 MMQ launch bounds — 2026-09-10

The active repacked Q6 MMQ row-64 and opt-in row-128 kernels now carry the
same `__launch_bounds__(256, 1)` contract. The focused varied-payload GPU test
passes. With the Q4 affine launch-bounds change already enabled, exact P512
measured `45.14` and `44.59 tok/s` across two runs, versus `43.17` and
`39.68 tok/s` before the Q6 attribute. **KEEP**; the pair mean is approximately
`8.3%` above the Q4-only launch-bounds path, with unchanged VRAM. The combined
path is still far below the 222.64 tok/s reference.

### Candidate — cached Q8_1 affine sum — 2026-09-10

The repacked affine kernels were recomputing the integer activation sum with
eight additional `sdot4` operations for every 128-value slab and output. The
Q8_1 MMQ quantizer now materializes the canonical half-rounded `d * qsum`
value once per group in `qsum_scaled`; the affine Q4/Q5 row-64 and row-128
kernels consume that value directly. The original float group sum remains in
`s` for compatibility, so the initial candidate block is 176 bytes.

The release build, all 11 host-only tests, and the focused gfx906 varied-Q6
GPU parity test pass. Exact Qwen3.8-27B-Q4_K_M P512 measured `50.24` and
`51.01 tok/s` (`10,190.86 ms` and `10,038.00 ms`) on two runs, versus
`45.14` and `44.59 tok/s` for the immediately preceding Q4+Q6 launch-bounds
candidate: `+12.8%` by pair mean. Allocation remained
`23,156,230,484` bytes. **KEEP** as the default M23 affine path; the target
`222.64 tok/s` remains unmet.

### Re-evaluation — compact cached-sum block — 2026-09-10

Removing the unused original `s` field and its reduction restored the block to
160 bytes and reduced the model allocation to `23,138,404,692` bytes. Focused
GPU parity passed, but exact P512 measured `46.70` and `47.41 tok/s`, versus
the 176-byte cached-sum control at `50.24` and `51.01 tok/s`. The compact
layout is **REJECTED**; the 176-byte block and its throughput are restored.

### Rejected experiment — row-128 × B128 tile geometry — 2026-09-10

The pinned mx configuration uses `I=128`, so the opt-in row-128 kernels were
changed to process 128 tokens per block instead of their existing 64-token
split. Focused varied-Q6/Q4 GPU parity still passed, but exact P512 dropped
to `39.43 tok/s` (`12,985.25 ms`) versus the cached-sum row-64 path at a
50.6 tok/s pair mean. The larger activation tile and doubled per-thread
accumulator footprint lose to register/working-set pressure on gfx906.
**REJECTED**; the source and launch geometry are restored to row-128 × B64
for the opt-in variant, leaving row-64 × B128 as the default.

### Rejected experiment — Q4 dead high-plane load removal — 2026-09-10

The Q4 affine MMQ tile has no high-bit plane, so a Q4-only specialization was
tested to skip its load and unpack work while retaining the same expanded LDS
mapping. Focused GPU parity passed, but exact P512 measured `48.16` and
`49.12 tok/s` (mean `48.64`) versus the cached-sum control mean `50.63 tok/s`.
The change is **REJECTED** and the generic validated reader is restored.

### Rejected experiment — Q4 affine occupancy-2 launch bounds — 2026-09-10

Because the Q4 row-64 kernel uses roughly 30.5 KiB of LDS, it was tested with
`__launch_bounds__(256, 2)` to invite two resident workgroups per CU. Focused
GPU parity passed. Three exact P512 runs measured `50.64`, `52.10`, and
`47.90 tok/s` (mean `50.21`) versus the cached-sum occupancy-1 pair mean of
`50.63 tok/s`; the result is not a repeatable gain and the third run regressed.
**REJECTED**; the pinned `__launch_bounds__(256,1)` contract is restored.

### Revalidation — restored 176-byte activation block — 2026-09-10

After the compact 160-byte experiment was rejected, the active block layout
was restored to retain both the original float group sums and cached
half-rounded `d * qsum`. A fresh exact P512 run with all M23 switches enabled
measured `10,123.01 ms / 50.58 tok/s`, with `23,156,230,484` allocated bytes.
This agrees with the earlier 50.24/51.01 tok/s pair and confirms the active
throughput result; the 222.64 tok/s target and model-sized numerical parity
remain unmet.

### Candidate — row-128 affine launch bounds — 2026-09-10

The opt-in row-128 affine kernel now carries the same gfx906
`__launch_bounds__(256,1)` contract as the active row-64 kernel. Focused
varied-payload GPU parity passes. Exact P512 with all M23 switches enabled
measured `8,942.30 ms / 57.26 tok/s` and `8,645.42 ms / 59.22 tok/s` across
two runs, for a `58.24 tok/s` mean versus the fresh row-64 `50.58 tok/s`
control. Allocation remained `23,156,230,484` bytes. **KEEP as an opt-in
row-128 configuration**; do not make it the default until model-sized parity
is qualified. The 222.64 tok/s target remains unmet.

### Rejected experiment — row-128 activation reuse loop order — 2026-09-10

The row-128 kernel was briefly reordered so each token's activation scale and
cached affine sum were loaded once and reused across both 64-row halves. The
focused varied-payload GPU parity test still passed, but exact P512 measured
`56.28 tok/s` after the earlier `57.26`/`59.22 tok/s` launch-bounds pair
(58.24 tok/s mean). The loop reorder is **REJECTED** and has been reverted;
the independent row-128 launch-bounds candidate remains the opt-in KEEP.

### Rejected experiment — packed-Q4 row-128 reader — 2026-09-10

An opt-in Q4-only row-128 specialization kept packed nibbles in LDS and
masked them at `sdot4` time, matching mx's direct packed-dot arithmetic. The
focused varied-payload GPU parity test passed. Exact P512 measured
`58.07 tok/s` and `52.24 tok/s` (mean `55.15 tok/s`) versus the expanded
row-128 launch-bounds pair at `58.24 tok/s` mean. The packed reader is
**REJECTED** for this workload and has been removed; the expanded reader
remains the opt-in KEEP.
