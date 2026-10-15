# EXP-0458 — Preserve GQA6 and combine each four-key tile

## Decision

The existing Q8-query dot4 route is not eligible for another GPU run. EXP-0409
already failed the real 64K full-model correctness gate for both single-plane
and one-residual-plane queries; EXP-0428 failed the same gate for four planes
at 8K. A restored 73,099+1,024 comparison would not change that product-level
decision. No GPU work or timing claim is made here.

The graph-path source audit found that dot4 selection accepts a non-null
`prefill_state` even when graph capture supplies host `base_position == 0`,
passes that state into the kernel, and includes the dot4/query-residual/ILP
flags in suffix-graph cache identity. This is source evidence only; no new
runtime route trace was collected.

## Proposed next dataflow

Production GQA6 Stage1 already launches one CTA per `(token, split, KV head)`
with block `(64, 3)`. Each of its three Wave64 groups owns two Q heads, so one
CTA retains all six Q heads for its KV head. The direct-global specialization
uses no LDS and already computes four independent QK scores and holds the
corresponding four V vectors per microtile (`UNROLL=4`). Its remaining serial
dependency is the four repeated online-softmax and PV updates.

The candidate will preserve that geometry and Q8 arithmetic. For each existing
four-key microtile, replace the sequential per-key `(m, l, acc)` recurrence
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

## Pre-build resource estimate at the restored workload

For a 1,024-token suffix with 24 query heads and 32 splits:

| Resource | Current direct-global control | Proposed candidate |
| --- | ---: | ---: |
| Stage1 CTAs | `1024 × 32 × 4 = 131,072` | same |
| Threads per CTA | `64 × 3 = 192` | same |
| Query heads per CTA | 6 | same |
| Dynamic LDS | 0 B | 0 B expected |
| Partial workspace | `1024 × 24 × 32 × 258 × 4 = 811,597,824 B` (774 MiB) | same |
| VGPR | existing compiler report required | estimate +0–15 over control; hard stop on spills |

The VGPR delta is only a source-level estimate: the four QK scores and four
Q8 V fragments are already live in the production unroll. The candidate adds
four normalized weights and a few scalar combine values, which may reuse dead
QK temporaries. Compile metadata for the exact gfx906 specialization is the
gate; do not infer occupancy from this estimate.

The fixed 32 splits distinguish this from EXP-0423, which doubled partitions
and was 0.3% slower with changed exact logits/state. The unchanged CTA grouping
distinguishes it from EXP-0427's larger query tiles, which were 6.6% slower
(tile 8) and 61% slower (tile 4). No sweep is proposed.

## Gates if implemented

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

No implementation or runtime result is claimed by this planning record.
