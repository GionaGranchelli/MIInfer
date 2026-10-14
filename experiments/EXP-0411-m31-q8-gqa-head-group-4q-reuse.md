# EXP-0411 — Q8 GQA-head grouping with four query rows

**Status:** REJECT at the P8K screen; production dispatch unchanged.
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

## Build/resource result

Built `miinfer` from commit `794f7d037fb00d632cd01a92caca6d1ad717734a` with
`localhost/miinfer-dev:rocm-7.2.1`
(`sha256:447907c697fb2c39ce9657401e354a70753d5672ad955d4ceb12de976b91224a`),
HIP 7.2.53211 and AMD clang 22.0.0git for `gfx906`. The HIP resource remark
reports for both `<true>` and `<false>` specializations: VGPR 50, SGPR 57,
scratch 0 bytes/lane, dynamic stack false, SGPR spills 0, VGPR spills 0, LDS
2,112 bytes/block, and compiler-estimated occupancy 4 waves/SIMD. The candidate
launch is 768 threads (12 Wave64s), below the gfx906 1,024-thread limit. Build
completed; no GPU run has started.

An offline mapping assertion enumerated 96 `(KV head, query row, query head)`
states: all four rows and 24 heads were covered exactly once, and every query
head mapped to its expected KV head.

After GPU2's guarded 64K control exits, run production then candidate on that
same selector and require byte-identical logits before proceeding:

```bash
/tmp/m31-q8-kvhead-4q-reuse/scripts/run-m31-q8-4q-screen.sh q8-gqa4-prod-p2k 0 2048 240 2 && \
/tmp/m31-q8-kvhead-4q-reuse/scripts/run-m31-q8-4q-screen.sh q8-gqa4-candidate-p2k 2 2048 240 2 && \
cmp /tmp/m31-q8-kvhead-4q-reuse/evidence/q8-gqa4-prod-p2k.f32 \
    /tmp/m31-q8-kvhead-4q-reuse/evidence/q8-gqa4-candidate-p2k.f32
```

## GPU screening result

Both matched checks ran on Machinist GPU2, BDF `0000:88:00.0`, in the pinned
image with the same Qwen3.8-27B-Q4_K_M model, tokenizer-backed prompt, Q8/Q8 KV,
and 65,536-token capacity. Each timing is one screening sample. All logits were
finite and byte-identical within each pair.

| Tokens | Path A | A prefill | Path B | B prefill | B vs A | Logits SHA-256 | Peak junction |
| ---: | --- | ---: | --- | ---: | ---: | --- | ---: |
| 2,048 | Production | 10.045 s | Grouped GQA | 9.677 s | 1.038x faster | `6540fa47b0c88586b1fcc46de8fae25837a4ed051e292da39edcbac1e17f8d80` | 53 / 54 C |
| 8,192 | Existing 4Q | 45.507 s | Grouped GQA | 48.598 s | 6.8% slower | `13d62d4b8f9a5869879e00ce875081a1df445ac460e18795f770abc9d78a431b` | 61 / 62 C |

The candidate passes exact P2K and P8K logits, but fails the predeclared P8K
performance gate. Do not run it at 64K. Keep the selector opt-in for any later
reconsideration; the grouped topology is rejected for this campaign. The small
P2K timing difference is a single sample and does not offset the P8K regression.

Raw logits, runner logs, and guarded temperature samples are committed under
[`results/EXP-0411/raw`](../results/EXP-0411/raw/). The GPU2 production 64K
control remains a distinct result: 1,883.543 s, finite logits, SHA-256
`9a33d3f3b190126806dee63074189df0f83b78503eac9f2f80e881babb16f482`.

If the P2K comparison is byte-identical, compare the candidate's full-model
P8K time and logits against the existing four-row control on that same device:

```bash
/tmp/m31-q8-kvhead-4q-reuse/scripts/run-m31-q8-4q-screen.sh q8-gqa4-4q-control-p8k 1 8192 600 2 && \
/tmp/m31-q8-kvhead-4q-reuse/scripts/run-m31-q8-4q-screen.sh q8-gqa4-candidate-p8k 2 8192 600 2
```
