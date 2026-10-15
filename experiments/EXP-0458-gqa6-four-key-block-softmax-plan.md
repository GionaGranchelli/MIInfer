# EXP-0458 — Preserve GQA6 and combine each four-key tile

## Earlier dot4 decision

The existing Q8-query dot4 route is not eligible for another GPU run. EXP-0409
already failed the real 64K full-model correctness gate for both single-plane
and one-residual-plane queries; EXP-0428 failed the same gate for four planes
at 8K. A restored 73,099+1,024 comparison would not change that product-level
decision. The dot4 route was excluded from the work below.

The graph-path source audit found that dot4 selection accepts a non-null
`prefill_state` even when graph capture supplies host `base_position == 0`,
passes that state into the kernel, and includes the dot4/query-residual/ILP
flags in suffix-graph cache identity. This is source evidence only; no new
runtime route trace was collected.

## Candidate dataflow

Production GQA6 Stage1 already launches one CTA per `(token, split, KV head)`
with block `(64, 3)`. Each of its three Wave64 groups owns two Q heads, so one
CTA retains all six Q heads for its KV head. The direct-global specialization
uses no LDS and already computes four independent QK scores and holds the
corresponding four V vectors per microtile (`UNROLL=4`). Its remaining serial
dependency is the four repeated online-softmax and PV updates.

The candidate preserves that geometry and Q8 arithmetic. For each existing
four-key microtile, it replaces the sequential per-key `(m, l, acc)` recurrence
with one block-online-softmax combine:

```text
m_tile = max(score[0..3])
m_new = max(m_old, m_tile)
alpha = exp(m_old - m_new)
w[i] = exp(score[i] - m_new)
l_new = alpha * l_old + sum(w[i])
acc_new[d] = alpha * acc_old[d] + sum(w[i] * v_scale[i] * v_q8[i,d])
```

The four QK computations stay independent, all six query heads remain in the
same CTA, and each query keeps its original causal range and 32-way split
interval. The tail loop, Stage2 split combine, state layout, KV storage,
production default, and graph replay contract remain unchanged. This changes
the online-softmax/PV dataflow within the existing key tile; it does not add
cross-token reuse or change tile/split geometry.

## Resource metadata at the restored workload

For a 1,024-token suffix with 24 query heads and 32 splits:

| Resource | Current direct-global control | Implemented candidate |
| --- | ---: | ---: |
| Stage1 CTAs | `1024 × 32 × 4 = 131,072` | same |
| Threads per CTA | `64 × 3 = 192` | same |
| Query heads per CTA | 6 | same |
| Dynamic LDS | 0 B | 0 B expected |
| Partial workspace | `1024 × 24 × 32 × 258 × 4 = 811,597,824 B` (774 MiB) | same |
| VGPR | 60 | 60 |
| SGPR | 50 | 48 |
| spills / private bytes | 0 / 0 | 0 / 0 |
| static LDS | 0 B | 0 B |

The metadata is from the exact gfx906 specialization. Both control and
candidate use a 192-thread workgroup. The candidate uses two fewer SGPRs, the
same 60 VGPRs, and no spill, private-segment, or LDS storage. This clears the
compiler resource gate; it does not establish runtime occupancy.

The fixed 32 splits distinguish this from EXP-0423, which doubled partitions
and was 0.3% slower with changed exact logits/state. The unchanged CTA grouping
distinguishes it from EXP-0427's larger query tiles, which were 6.6% slower
(tile 8) and 61% slower (tile 4). No sweep is proposed.

## Predeclared gates

- **Compiler:** pinned gfx906 image, exact direct-global specialization;
  record VGPR, SGPR, spills, private bytes, LDS, workgroup and wave size.
  Reject immediately on spills, unexpected scratch/LDS, or a wrong target.
- **Oracle:** deterministic four-score/four-value FP32 reference, including
  old-max-wins, new-max-wins, ties, negative-infinity initialization, and
  Q8 V scales. Compare `m`, `l`, partial accumulator and normalized output.
- **Runtime correctness:** same-GPU2 restored 73,099-token prefix and
  tokenizer-backed 1,024-token suffix; verify exact route trace on all 16 GQA
  layers, final position 74,123, nonempty recurrent trace, finite logits/state,
  `max_abs <= 0.01`, cosine `>= 0.9999`, and generated-token agreement.
- **Performance:** one direct-global/candidate pair on GPU2, same pinned image,
  binary, model, session and prompt. Do not repeat or run cold 64K unless the
  correctness gate passes and the candidate is at least 10% faster.
- **Safety:** verify GPU2 BDF `0000:88:00.0`; warn at 80 C and stop at 85 C.

## Implementation and result

The opt-in kernel and pair harness were implemented and pushed on branch
`perf/m31-gqa6-attention-rewrite-v2-gpu2`. The production default remains off.
The independent FP32 oracle passed four edge cases and 128 accumulated
four-key tiles. The full target compiled in the pinned image.

The exact gfx906 code object was
`/tmp/m31-gqa6-attention-rewrite-v2-gpu2/build/mi50-release/CMakeFiles/miinfer_hip.dir/gfx906/kernels/qwen3_primitives.hip.o.0.hipv4-amdgcn-amd-amdhsa--gfx906`
on Machinist (SHA-256
`737a733664e58e752d8b1848e7368b23c2b5efd38538f94394f4966d869c56f4`). The
resource excerpt is preserved in
[`stage1-resource-metadata.txt`](../results/EXP-0458-gpu2-pair-1024/stage1-resource-metadata.txt).
The tested image was `localhost/miinfer-dev:rocm-7.2.1`, image ID
`50ff28a28cc13a3ca7333cd0c70e322d16245967c5fdf0d30a1bf57bda80a512`, repo
digest `sha256:447907c697fb2c39ce9657401e354a70753d5672ad955d4ceb12de976b91224a`.

### Graph timing pair

One matched pair ran on GPU2 BDF `0000:88:00.0` (`gfx906`). Control and
candidate used the same 73,099-token GPU2 checkpoint, tokenized prompt, model,
and pinned image. The prefix plus suffix reached position 74,123. Route traces
showed direct-global in all 16 control GQA layers and block-softmax4 in all 16
candidate GQA layers, with graph state active. Both returned the same first two
tokens (`18369`, `198`) and finite outputs.

| Route | Restore | 1,024-token suffix | Position | First / second token |
| --- | ---: | ---: | ---: | --- |
| Direct-global control | 2.856 s | 20.888 s | 74,123 | 18,369 / 198 |
| Block-softmax4 candidate | 2.928 s | 20.303 s | 74,123 | 18,369 / 198 |

The candidate was only `1.029x` faster, below the predeclared `1.10x` gate.
It also failed numerical acceptance: logits max-abs `0.733937979`, cosine
`0.999650309`; recurrent state max-abs `6.523668110`, cosine `0.993032655`
across 48 recurrent layers. Token equality and finite outputs do not override
those failures. No repeat or cold-64K run was made.

The raw run log contains a stale summary tag `flash_tile_pair=1`; that label
was inherited from the earlier comparator. The 16 direct-global and 16
block-softmax dispatch traces, plus the route-specific result label, identify
the actual tested candidate. The source harness label was corrected after
this run.

### Non-graph layer localization

To check for a local kernel or mapping defect, one additional bounded
non-graph restored run compared the candidate attention output against
direct-global at each of the 16 GQA layers. For each shadow comparison it
saved the candidate gated output, reran the same layer on the same Q/K/V and
cache state with direct-global, measured the outputs, restored the candidate
output, and continued the session. The run reached position 74,123 and
observed all 48 recurrent layers. It is diagnostic only; its non-graph wall
times are not used for performance.

| GQA layer | Output max-abs | Cosine | Finite |
| ---: | ---: | ---: | :---: |
| 3 | 0.000005484 | 1.0 | yes |
| 7 | 0.000018716 | 1.0 | yes |
| 11 | 0.000012159 | 1.0 | yes |
| 15 | 0.000013113 | 1.0 | yes |
| 19 | 0.000032187 | 1.0 | yes |
| 23 | 0.000034094 | 1.0 | yes |
| 27 | 0.000057221 | 1.0 | yes |
| 31 | 0.000030041 | 1.0 | yes |
| 35 | 0.000038624 | 1.0 | yes |
| 39 | 0.000035286 | 1.0 | yes |
| 43 | 0.000030756 | 1.0 | yes |
| 47 | 0.000059128 | 1.0 | yes |
| 51 | 0.000087738 | 1.0 | yes |
| 55 | 0.000062943 | 1.0 | yes |
| 59 | 0.000141144 | 1.0 | yes |
| 63 | 0.000189781 | 1.0 | yes |

All per-layer attention outputs were finite and closely aligned with
direct-global, with the largest local max-abs difference at the final GQA
layer. Yet the complete candidate session again had logits max-abs `0.73394`
and recurrent state max-abs `6.52367`. This localizes the failure to the
accumulated effect of changed FP32 association through the full recurrent and
attention stack, rather than a large single-layer scale, head, or boundary
error. The measured drift means this rewrite is not acceptable for MIInfer's
current correctness contract.

The graph pair peaked at 55 C and the layer-localization run at 60 C; neither
reached the 80 C warning or 85 C stop threshold. The exact model SHA-256 was
`7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`; the
prompt SHA-256 was
`fa4a0de5db813f5f06b58de5b4b7df8f234c4c808e0965bd44c545e9087c462c`; the
GPU2 session SHA-256 was
`bbbe9543d7ccbb740cc3db1cda7bc959e8e95926587a31b9cb8e3510c557de96`.
The first graph-pair binary SHA was
`d73cfdd99f6d8bc3be8811e82b2df7f9be08c44c945d0dfc4bbc0b1deafd251b`; the
later non-graph diagnostic binary, rebuilt with the layer-shadow probe, SHA
was `393d8119f4b790df1b5eef53c345d53ce425e54f174f7b2da21abc027be6259e`.

Raw logs and telemetry are preserved under
[`results/EXP-0458-gpu2-pair-1024`](../results/EXP-0458-gpu2-pair-1024) and
[`results/EXP-0458-gpu2-layer-localization-1024`](../results/EXP-0458-gpu2-layer-localization-1024).
The diagnostic was compiled with the same pinned image after adding the
per-layer comparison hook. The harness now uses one environment variable,
`MIINFER_M31_Q8_GQA6_BLOCK_COMPARE_LAYERS`, to disable graph capture for and
enable that probe.

**Decision: reject this schedule for performance qualification.** Keep the
opt-in implementation and report on this isolated research branch for
reproducibility; it must not become the production default. The next schedule
must preserve production's per-key FP32 update order while seeking instruction
overlap or reduced load/dependency latency.
