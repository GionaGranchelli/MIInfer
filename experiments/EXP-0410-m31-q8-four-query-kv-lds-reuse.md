# EXP-0410 — Q8 four-query KV-tile LDS reuse

**Status:** SHORT GATES PASS; 64K CANDIDATE COMPLETED; SAME-BDF CONTROL IN PROGRESS  
**Baseline:** `f91fd42072113108a9bfa5c1afd96e989ef29a2a`

## Gate definition

- **Objective:** Reduce Q8/Q8 late-context prefill time on gfx906 by reusing the same K/V data across nearby causal query rows.
- **Current full-model bottleneck:** EXP-0398 measured Q8/Q8 causal attention at 2,644.759 ms of a 4,944.550 ms late 512-token tile at P8192 (53.488% of tile time; 83.68% of GQA block time). EXP-0404 also attributes 58.448% of the FP16-KV late tile to attention. Attention is therefore a material target, though those profiles do not predict candidate speedup.
- **Reference:** Production Q8 Split-K stage 1 in `gfx906/kernels/qwen3_primitives.hip`; 32 splits, one query token × a two-head pair per CTA, direct Q8 K/V, sequential online softmax. EXP-0405's 16-query Q8 route staged/dequantized fragments and altered intermediate precision; its full-model logits failed (`max_abs=0.657086`, cosine `0.999395`).
- **Difference being tested:** One 256-thread CTA handles 4 adjacent query tokens × one two-head pair × one existing split. It stages each 4-key Q8 K/V tile and scales once in LDS, then reuses it for the eight query/head rows. Preserve each row's exact split bounds, QK reduction tree, online-softmax update order, value accumulation order, and production stage 2. Only CTA/data movement changes.
- **Rejected families distinguished:** Unlike EXP-0360, this is the existing Q8/Q8 Split-K route, not six-wave FP16 full attention. Unlike EXP-0362/0363, it keeps only four query-token states per CTA, has no 16-token GQA2 fragment schedule, and does not change the production arithmetic or softmax phase decomposition. It is a bounded Q8 KV-load reuse schedule, not a geometry transplant.
- **Expected maximum end-to-end impact:** The attention share above is only an Amdahl ceiling; no speedup is assumed.
- **One independent variable:** Stage-1 CTA grouping and K/V data movement; fixed model, prompt, Q8 cache, 32 splits, arithmetic, output format, tile size, and stage 2.
- **Resource gate:** Pinned ROCm 7.2.1 / `localhost/miinfer-dev:rocm-7.2.1`; gfx906 metadata must show zero spills/private scratch and LDS below 64 KiB. Use Machinist GPU0 (`0000:04:00.0`, unique `0x21678e17348c2f7`), verify process placement, and maintain 80/85 C warning/stop.
- **Correctness gate:** On identical tokenizer-backed 2K full-model input, finite outputs, `max_abs <= 0.01`, cosine `>= 0.9999` against production Q8. Do not time candidate until this passes.
- **Performance gate:** One 8K matched full-model screen after 2K parity; same host, device, model, prompt prefix, capacity, and environment. Proceed to exact 64K only if parity passes and candidate is faster at 8K.
- **Kill criteria:** Any HIP fault, invalid/non-finite result, mismatch over tolerance, compiler spill/private scratch, thermal stop, or slower 8K result.
- **Success criteria:** Correct 2K and 8K results, clean resources, measured 8K full-model improvement; then exact 65,536-token cold prefill A/B.
- **If rejected:** Keep production dispatch unchanged; record measured failure and stop this dataflow family.

## Evidence

The primary matched controls and candidates were built from this experiment's
f91 source with the pinned ROCm 7.2.1 container. Both use model
`Qwen3.8-27B-Q4_K_M.gguf`, prompt `m31-64k-context.txt`, Q8/Q8 KV, capacity
65,536, GPU0 (`0000:04:00.0`, unique `0x21678e17348c2f7`), and identical
environment except the candidate selector. Runs were one sample per condition;
the timings are screening evidence, not a repeated performance claim.

| Tokens | Production prefill | Candidate prefill | Candidate speedup | Output comparison | Peak junction |
| ---: | ---: | ---: | ---: | --- | ---: |
| 2,048 | 9.329 s | 8.740 s | 1.067x | identical SHA `6540fa47b0c88586b1fcc46de8fae25837a4ed051e292da39edcbac1e17f8d80`; finite | 58 / 57 C |
| 8,192 | 54.567 s | 41.993 s | 1.299x | identical SHA `13d62d4b8f9a5869879e00ce875081a1df445ac460e18795f770abc9d78a431b`; finite | 69 / 69 C |

Compiler resource remark for the candidate specialization:

```text
TotalSGPRs=57, VGPRs=50, scratch=0 bytes/lane, SGPR spill=0,
VGPR spill=0, LDS=2112 bytes/block, occupancy=4 waves/SIMD.
```

The exact 65,536-token cold candidate completed on GPU0 under the 80/85 C
thermal guard. It processed 65,536 tokens in 867.841 s, produced 248,320
finite logits (SHA-256
`9a33d3f3b190126806dee63074189df0f83b78503eac9f2f80e881babb16f482`), and
peaked at 76 C junction. The production control on the same BDF is running;
no 64K speedup or output-equivalence claim is made until it completes.

## Exact 64K reproduction commands

Candidate (completed):

```bash
/tmp/m31-q8-lds-gpu0/scripts/run-m31-q8-4q-screen.sh q8-4q-lds-candidate-64k 1 65536 4000
```

Same-BDF production control (currently running):

```bash
/tmp/m31-q8-lds-gpu0/scripts/run-m31-q8-4q-screen.sh q8-4q-production-64k 0 65536 4000
```

Both use the same release binary, image digest
`sha256:447907c697fb2c39ce9657401e354a70753d5672ad955d4ceb12de976b91224a`,
model path, 65,536-token prompt, Q8/Q8 path, and GPU0 selector. The candidate
alone sets `MIINFER_M31_Q8_4Q_LDS_REUSE=1`. The completed candidate's output,
timing, and GPU0 telemetry are retained under
`/tmp/m31-q8-lds-gpu0/evidence/` and on Machinist at
`/home/machinist/m31-prefill-0002-20261014/`.

An independent production control also completed on GPU2 (`0000:88:00.0`,
unique ID `0x2152414173497dd3`): 65,536 processed tokens, 1,883.543 s prefill,
finite 248,320 logits. Its full-logit SHA-256 is the same
`9a33d3f3b190126806dee63074189df0f83b78503eac9f2f80e881babb16f482` as the
GPU0 candidate. This establishes exact output agreement across these two
runs. Since they used different physical cards, their elapsed times are not a
matched speedup result; the same-BDF GPU0 production control remains the
performance comparison.
