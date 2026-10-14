# EXP-0446 — GQA6 attention dataflow audit

**Result:** no new kernel candidate admitted; source and runtime unchanged.
**Base:** `28e467f07704985ec02486c553da4300a63615d8` (direct-global GQA6 report)
**Scope:** Machinist MI50 GPU2 review; no GPU benchmark was run for this audit.

## Current execution shape

`qwen35_splitk_suffix_attn_stage1_quant_kernel<true,true,4,true,false,false,true>`
launches one CTA per query token, Split-K partition, and KV head. Each CTA has
three Wave64s; each wave computes two query heads in its 32-lane halves. Each
lane handles eight head dimensions. Four K/V rows are loaded per loop group,
then each score is reduced and each head's online-softmax/value state is
updated in key order. Split-K partials are combined by a separate Stage2.

The best current alternative is direct-global Q8 K/V: it removes LDS
staging/barriers, preserves the arithmetic and exact results, and improved a
restored 73,099 + 1,024 suffix by 7.8% (18.805 s versus 20.405 s median).
This remains below the 10% retain threshold and does not establish a cold
64K result.

## Closed candidate shapes

The repository already contains real MI50 evidence against repeating these
nearby variants:

| Change | Evidence | Decision |
| --- | --- | --- |
| Cross-token/query tiles 2–8 | Exact tile 2 gained under 1%; tile 4 was 61% slower and tile 8 6.6% slower. The logical reuse census confirms query split bounds are not the blocker. | Do not sweep another tile size. |
| UNROLL 8 | Exact, but 24–26% slower in the matched packed-U128 screen. A separate earlier U8 route reported 58 VGPR spills. | Keep U4. |
| One wave serially updates all six GQA heads | 38.402 s versus 19.031 s and failed full logits/state. | Reject. |
| Block/stable softmax reductions | EXP-0442 diagnosed an actual layer-3 mapping failure and 2.08x slowdown; EXP-0445 was only 4.1% faster once and failed logits/state by wide margins. | Do not tune reduction grouping without a new numerical oracle. |
| Q8 query fragment reuse / dot4 / FP16 dot2 | Some routes were faster, but failed full-model logits or recurrent-state correctness; FP16 dot2 was also 6–7% slower. | Do not trade current Q8 trajectory correctness for isolated speed. |
| Wave64 half shuffle for duplicate loads | Compiler emitted the guarded loads anyway and added shuffle work. | Rejected before GPU execution. |

The measurements have different source/image revisions and workloads as stated
in their individual reports; this table is a design-space index, not a pooled
benchmark.

## What a genuine rewrite must change

Another tile-size, unroll, or scalar online-softmax regrouping is not a new
approach. The next materially different design is a FlashAttention-style
kernel that computes a multi-query × key tile and reuses each Q8 K/V tile
across query rows while maintaining per-query online-softmax state. It needs
an explicit numerical oracle before the long-context run because the existing
query-tiled implementation was faster but diverged, and the key-block
reduction implementation contained a mapping error.

Such a rewrite must first prove, on a small fixed operand capture, that each
query row uses its own causal/split interval, Q8 scales, GQA head mapping, and
the expected output/state layout. Only after that should it build for gfx906,
show no spills or unintended workspace, and enter one restored-session A/B.
The preserved-Q8 arithmetic contract and the faster tile-level reduction
contract may conflict; resolve that in the candidate design rather than
claiming equivalence based on matching generated tokens.

## Disposition

No source or production-default change is made. GPU2 was idle at 25 C during
preflight; this audit did not run inference or disturb either GPU. Continue
from direct-global GQA6 as the safe comparison path. Do not repeat the closed
variants above. A future FlashAttention-style rewrite should be admitted only
with a concrete per-row mapping, split-bound proof, and numerical comparison
plan.
