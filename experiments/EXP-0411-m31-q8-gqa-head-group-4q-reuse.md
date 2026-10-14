# EXP-0411 — Q8 GQA-head grouping with four query rows

**Status:** IMPLEMENTED; build/resource gate pending; no GPU run started.
**Base:** `f50abbf` on `codex/m31-q8-lds-gpu0`.

## Hypothesis

The EXP-0410 four-row CTA stages each KV tile once per two-query-head pair.
Qwen3.8 has six query heads per KV head, so three such CTAs still load the
same KV tile. Group those three pairs into one CTA, retaining the four-row
reuse and each query/head's independent online-softmax state. This can reduce
Q8 K/V tile reads by up to 3x within the EXP-0410 mapping; synchronization,
larger CTA allocation, and register residency may erase that benefit.

## Evidence and reference

EXP-0398 attributed 53.488% of the Q8/Q8 late P8192 tile to causal attention.
EXP-0410's current 4Q mapping passed exact-logit 2K and 8K screens and measured
1.299x at 8K in one sample. These establish an attention target and a viable
control, not an expected speedup for this candidate. The theoretical late-tile
ceiling is 2.15x if all measured attention time vanished; whole-prefill impact
will be lower and must be measured.

Strata PRs #1661 and #1718 were considered. Their query LDS swizzle and
12-head reduce-scatter do not match this kernel's Q8 KV staging or independent
per-query/head softmax outputs, so no code or reduction pattern was copied.
This candidate reuses the local EXP-0410 kernel only; no external source code
was borrowed.

## Candidate

One independent variable: CTA mapping. A compile-time specialization maps one
CTA to one KV head, four adjacent tokens, all three query-head pairs, and one
of the existing 32 splits. It uses 768 threads (12 Wave64s). Each 32-lane
head-state retains the same Q load, shuffle reduction, online-softmax update
order, and output slot; the existing stage 2 is unchanged. The KV tile is
loaded once by the CTA and read by all 24 row/head states.

Opt-in selector: `MIINFER_M31_Q8_GQA_4Q_LDS_REUSE=1`. The existing
`MIINFER_M31_Q8_4Q_LDS_REUSE` selector and default production dispatch remain
separate and unchanged when the new selector is unset.

## Gates

- **Resource:** gfx906 compiler metadata must show zero VGPR/SGPR spills, zero
  private scratch, LDS below 64 KiB, and a legal 768-thread workgroup. Stop if
  the compiler spills or rejects the mapping.
- **Correctness:** On the same pinned MI50 and exact tokenizer-backed 2K
  workload as EXP-0410, compare production Q8/Q8 logits; require finite values,
  max absolute error <= 0.01, and cosine >= 0.9999 before timing.
- **Performance:** Compare with EXP-0410's 4Q path on the same GPU and workload
  at 8K. Promote only if correct and faster; then run the exact 65,536-token
  cold prefill against the matched production control.
- **Kill:** compiler resource failure, numerical gate failure, or slower 8K
  than the 4Q control. GPU timing is deferred until a guarded MI50 is free.
