# V2-0043 — Reference-shaped GQA attention rewrite

## Objective

Reproduce the proven KQ-fragment/KV-reuse dataflow of pinned mx and upstream
in MIInfer's Qwen3.8 gfx906 P8192 suffix-attention path. This is a structural
rewrite, not a geometry sweep. Keep it opt-in until the full promotion gate is
met.

## Baseline and bottleneck evidence

* MIInfer P8192 attention-family device work: 5,099.280 ms; total device work:
  39.645 s. The hard device-work ceiling is 12.86%; wall-time impact is lower
  because kernels overlap.
* MIInfer P8192 stage 1: one query position × a two-head pair per CTA, grid
  `12×512×3`, 64 threads, three KV splits; request window contains 240
  stage-1 and 240 stage-2 launches.
* Pinned mx attention-family device work: 1,784.926 ms; its confirmed P8192
  main tile is `flash_attn_tile<256,256,16,2,false>`, grid `{32,3,12}`,
  block `{32,8,1}`, followed by split combine.
* Pinned upstream recapture evaluated 8,193 tokens (8,192 rendered prompt
  tokens plus BOS) and confirmed the same main `<16,2>` kernel: 256 main calls
  and 288 combine calls. The 16-call `<4,2>` and `<2,2>` tails cover edge
  batches. Exact trace and source revisions are recorded in EXP-V2-0042.

## Reference mechanism reconstructed

Sources: pinned mx `2e9d29fe736969160f17476ecf0a6298cee6966` and pinned upstream
`73a43d1f69345aee8bb186ef4b3172cef892f2e5`, both
`ggml/src/ggml-cuda/fattn-tile.cuh`. For the observed `<16,2>` shape,
`ncols=32`, four warps, and each warp owns eight Q columns. Adjacent Q heads
share a KV head.

| Lifetime / work | Reference | MIInfer target |
|---|---|---|
| CTA query ownership | 16 tokens × 2 adjacent Q heads = 32 rows | same |
| Q lifetime | Q tile staged once in `Q_tmp` for CTA lifetime | load Q rows once and retain for CTA lifetime |
| K tile / KQ chunks | AMD config for `<16,2>`: `nbatch_fa=32`, `nbatch_K=128`; a 32-position K tile is processed in two 128-dimension chunks | match 32 K positions and two 128-dimension chunks; each K fragment is shared across four Q rows per logical warp |
| K load/reuse | each KQ dimension chunk stages K into `KV_tmp`, then the loaded fragment is reused across four Q columns; 2 dimension chunks per 256-d head | two 128-dimension K stages per 32-position tile, shared across the 32 Q rows |
| KQ representation | register dot accumulators become a compact tile-local KQ fragment in shared `KQ` | materialize tile-local KQ scores/weights once; do not recompute during V |
| online state | per-Q max/sum and register `VKQ`; rescale accumulator as max changes | per-row max/sum and output accumulator carried across KV tiles |
| V load/reuse | `nbatch_V=16`; KQ is consumed before `KV_tmp` is reused for 16-position V fragments, each feeding all four Q columns | reuse K workspace for 16-position V fragments after KQ; each V value feeds all four rows per logical warp |
| split/reduction | partials across split grid, separate combine kernel | write per-row split max/sum/accumulator, reuse existing separate combine |
| resource evidence | selected code object: 27,136 B LDS, 97 VGPR, 46 SGPR, no spills | iteration 7: 31,104 B LDS, 81 VGPR, 43 SGPR, no spills; LDS permits at most 2 CTAs/CU |

The exact MIInfer failure modes are established by the original experiment
records and implementation commits:

* EXP-0362 (`423945c`) copied BQ16/GQA2 geometry but kept eight Q/output state
  sets live per Wave64. It compiled with 137 VGPR spills and was 3.4–5.4×
  slower at P512–P8192. Geometry was not equivalent to the reference schedule.
* EXP-0363 (`58a8029`) removed spills in its final v2 (158 VGPR, zero spills,
  25 KiB LDS), but retained the wrong lifetime order: query-row work enclosed
  the KV-tile loop, restaging K and repeating V per row group. It remained
  ~5.3× slower at P8192. Spill elimination alone did not reproduce reuse.

## Candidate invariants

1. A CTA owns all 16 query positions × 2 Q heads for one KV head.
2. One K tile is staged from global memory once per CTA/split/tile. Every row
   consumes it from shared storage; no row-group performs another global K
   stage.
3. KQ is materialized for all relevant rows and is tile-local. V processing
   uses that fragment without recomputing QK.
4. The shared K workspace is reused for V only after all KQ/softmax consumers
   finish; one V staging serves every applicable row.
5. Compact online max/sum/output state survives across KV tiles. The split
   combine remains a separate kernel.

## Resource / ISA gate

Before runtime timing, require 256 threads, zero VGPR/SGPR spills, zero
private/scratch bytes, and LDS within the gfx906 per-workgroup budget. Record
the theoretical resident CTA/wave limit, but do not claim achieved occupancy
without runtime measurement. Inspect generated gfx906 ISA for scratch
traffic, global K/V load count, LDS access behavior, scalarization, waits, and
the intended FP16 KV / FP32 accumulation. A spill or pathological code object
stops the candidate before timing; then revise the state/lifetime layout from
that compiler evidence, not by changing tile dimensions.

## Correctness and performance gates

* Compare to the existing FP16 control at P512/P2048/P4096/P8192. Verify
  causal masks, 24 Q heads / 4 KV heads / D=256, 6:1 mapping, three-split
  semantics, no NaN/Inf, and the existing FP16 numerical envelope. Run a
  full-model greedy/token-parity check before promotion.
* Keep one clean control and one candidate. First performance test is the
  isolated P8192 attention A/B; losing P8192 candidates do not proceed to a
  full model curve.
* Promote to end-to-end P8192 only after at least a material 15% isolated
  attention win. Then compare the exact competitive request to pinned mx and
  upstream, capture request-window attention attribution, and check VRAM plus
  P512/P2048/P4096 regressions.
* Primary success: MIInfer P8192 end-to-end wall time equal to or below pinned
  mx under the same request contract. Device-work sums are not a substitute
  for wall time.
* Continue correcting the implementation from measured compiler/runtime
  evidence after a loss. Stop only after end-to-end competitive closure and no
  longer-dominant attention cost, or direct compiler/hardware evidence that
  this actual dataflow cannot be made competitive on gfx906.

## Iteration log

### Iteration 11 hypothesis / gates

The real P8192 activation comparison localized the first suffix call (layer 3,
base position 512) to FP32-Q-to-FP16 conversion: actual Q has max abs `11.6942`,
RMS `1.19865`, and max conversion error `0.0039053`; with identical real Q/K/V,
iteration 10 differs from control by max abs `8.38637e-5`, RMS `1.25813e-6`.
The 128-token greedy request first diverges at approximately generated token 67.

Test one variable: retain FP32 Q in LDS and use FP32×FP16 scalar FMA for QK,
keeping iteration 10's CTA ownership, split count, KQ tile, in-place FP32
weights, K/V reuse, and output accumulation unchanged. Expected correctness
effect: remove query-rounding drift. Resource gate: zero spills/private bytes;
FP32 Q staging raises LDS from 29,056 B to approximately 45,440 B (one CTA/CU
by LDS arithmetic). Correctness gate: exact 128-token greedy IDs. Performance
gate: five interleaved isolated P8192 pairs at 32 splits and ≥15% median win.
Kill if resources spill, token parity still fails, or isolated win misses the
gate. Expected end-to-end ceiling remains the measured 12.86% attention share.
If rejected, keep iteration 10 as a diagnostic-only finding, not promoted.

Iteration 11 result: resource/ISA gate passed at 86 VGPR / 48 SGPR, zero
spills/private bytes, and 45,440 B LDS (one CTA/CU by LDS arithmetic). Real
attention error fell to max abs `8.9407e-7` / RMS `7.98851e-9`, but the greedy
text still differed at 16 tokens. Five 32-split P8192 pairs had medians 468.901
ms candidate / 338.587 ms control (0.722×), so this scalar-FMA representation
is rejected. Its 94× real-output error reduction motivates only a compensated
FP16 representation that restores query precision while retaining `v_dot2`.

Iteration 12 result: resource/ISA gate passed at 77 VGPR / 48 SGPR, zero
spills/private bytes, and 45,440 B LDS (one CTA/CU by LDS arithmetic). The
two-term FP16 query expansion retained `v_dot2` but required a second QK dot
and one-CTA LDS occupancy. Five P8192 32-split pairs measured medians
895.534/338.445 ms (candidate/control; 0.378×); synthetic max abs error was
`3.72529e-9`. This representation is rejected without a full-model run.

### Iteration 13 result

Keeping the first chunk on control and switching suffix chunks to iteration
10's 3-split candidate did not preserve output: the 128-token run matched the
first five IDs, then diverged at generated token 6 (candidate `7701`, control
`5686`). The isolated 3-split win therefore does not pass model correctness.

### Iteration 14 hypothesis / gates

The candidate currently partitions splits using the final token's maximum
context length for the whole 16-token CTA. The actual control computes
`chunk_size = ceil((base_position + token + 1)/num_splits)` independently for
each query token. Reproduce that per-query partition while keeping the
reference-pinned 3 splits, tile geometry, Q precision, and online schedule
fixed. Stage only the union of those per-query split intervals, mask each row
to its exact control interval, and reuse overlap from the staged K/V tile.
Correctness gate: direct isolated comparison and exact 128-token P8192 greedy
IDs. Performance gate: five interleaved isolated P8192 pairs and ≥15% median
win. Do not repeat end-to-end timing if parity or isolated performance fails.

### Iteration 14 result

The kernel now uses the control's per-query causal split bounds. Isolated
FP16-KV comparisons at P512/P2048/P4096/P8192 were finite and matched the
control within `7.16e-9` max absolute error. Five interleaved P8192 pairs at
three splits had a 250.227 ms candidate median versus 348.829 ms control
median (1.394x; 28.3% less attention time), clearing the isolated gate.

The initial direct full-model greedy check used `MIINFER_CONTEXT_CAPACITY=16384`,
`MIINFER_V2_0043_GQA_ATTENTION=1`, `MIINFER_DUMP_TOKENS=1`, the same prompt
(`"hello " * 8192` followed by a long-form computing-history request), and
128 generated tokens. Prompt length was 8,216 tokens. Candidate and control
both generated 128 tokens with identical token-ID SHA-256
`9a1cf1b3a4cdb37eccab02c0bb724df3459f553111395c1738feb8f6c2f7e75d`.
One non-interleaved pair measured candidate/control prefill at
287,991.43/287,931.73 ms and decode at 4,647.36/4,642.83 ms. This is
effectively neutral end-to-end, not a demonstrated model-level win; junction
temperature reached 92°C during the candidate run. `rocm-smi` reported
SCLK 1606 MHz, HBM 1000 MHz, and 100% GPU use at the sample point; paired
clock/temperature telemetry was not captured for the control. The exact CLI
shape was:

```sh
MIINFER_CONTEXT_CAPACITY=16384 MIINFER_V2_0043_GQA_ATTENTION=1 \
MIINFER_DUMP_TOKENS=1 build/mi50-release/miinfer run \
  /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  --prompt "$(python3 -c 'print("hello "*8192 + "Ignore the repeated text. Write a detailed, long-form history of computing and continue the essay; do not conclude early.")')" \
  --max-tokens 128 --no-stream
```

The control used the identical command with `MIINFER_V2_0043_GQA_ATTENTION`
unset. A P8193 repeated-hello check also
matched all 37 naturally generated tokens, but stopped on EOS and is secondary
evidence only.

### Route-audit correction (2026-09-29)

The preceding direct-CLI P8216 parity/timing results did **not** exercise this
candidate. `miinfer run` executes `FullAttentionLayer` in
`tools/qwen35_gpu_pipeline.hpp`; the first opt-in selector had only been
connected to `PrefillV2AttentionLayer`. The equal 128-token hashes and neutral
timing above are therefore default-path observations, not candidate
qualification. The serving API's `PrefillV2AttentionLayer` path does select the
candidate, but currently rejects its reusable-prefill-state mode before launch.

The opt-in call is now connected to `FullAttentionLayer::finish_prefill_attention`,
the full-layer-major wide-prefill path used by the M26 configuration. A narrow
P1025 check under that configuration logged the actual layer-3/base-512/512
candidate call; it generated the same first token (`248068`) as control. The
single prefill samples were 6,294.53 ms candidate vs 4,948.28 ms control; this
small-context timing is not a performance conclusion.

The active-route P8216 greedy A/B completed 128 tokens but diverged first at
generated token 36. Candidate/control IDs had different hashes. Candidate
prefill/decode were 43,526.50/8,028.54 ms versus 51,972.65/10,196.94 ms
control in one sequential pair, a promising but unqualified timing direction.
At the first suffix attention call (layer 3, base 512, count 512), direct
candidate-vs-control output comparison measured `max_abs=4.97699e-5`,
`rms_error=5.07124e-7`, 20,141 exactly equal values of 3,145,728. Thus the
FP16-Q candidate is not correct enough for exact greedy parity and is not
promoted despite the one-pair speed direction.

### Iteration 15 result — FP32 Q

The corrected per-query split implementation with FP32 Q compiled at 83 VGPR,
48 SGPR, zero VGPR/SGPR spills, zero private bytes, and 45,440 B dynamic LDS
(one CTA/CU by LDS). Synthetic P512/P2048/P4096/P8192 outputs matched within
`3.73e-9` max absolute error. Five isolated P8192 pairs measured candidate
median 644.709 ms versus control 349.106 ms (`0.541x` control/candidate), so
the FP32-Q representation is rejected without an active-route model run. This
retests the prior rejected I11 under the corrected split contract and confirms
that its accuracy-oriented arithmetic is not viable for this prototype.

Current disposition: the structural FP16-Q prototype demonstrates a real
isolated P8192 win and one active-route end-to-end speed direction, but fails
exact token correctness; the corrected FP32-Q variant restores isolated
accuracy but loses the performance gate. Keep both opt-in, promote neither,
and do not run a full curve.

### Iterations 16–20 — precision and generated operand schedule

These iterations retained the same 16×2 CTA, 32-key tile, three splits,
per-query split bounds, and FP32 online/output state. Each was screened with
the P512/P2048/P4096/P8192 isolated correctness sweep before P8192 timing.

| Iteration | Single change / result | P8192 disposition |
|---|---|---|
| 16 | FP16 high+residual Q using a second `v_dot2` pass; synthetic max error `3.73e-9`. Five-pair median 875.654 ms candidate / 348.352 ms control; resource 80 VGPR / 48 SGPR, no spills, 45,440 B LDS. | REJECT — accuracy recovered, but 0.398× control/candidate; added QK arithmetic dominates. |
| 17 | Reload residual Q from global to restore Q staging to 16 KiB; synthetic max error `3.73e-9`. Median 809.124 / 348.919 ms; 95 VGPR / 48 SGPR, no spills/private bytes, 29,056 B LDS. | REJECT — 0.431×; reducing LDS by 16 KiB recovered only ~8%, so residual-dot work is the main cost. |
| 18 | Store only normalized weights as FP16 KQ, matching reference representation, while preserving float accumulation in V. Synthetic max error `2.59e-6`; median 331.204 / 349.657 ms. | REJECT — ~5.6% win, below 15%; scalar half-to-float weight use did not reproduce reference's packed half2 consumer. |
| 19 | Keep scores in registers through softmax; store/reload only tile-local float weights. Synthetic max error `7.16e-9`; median 329.565 / 349.028 ms. | LEARN — ~5.9% win; isolating score lifetime showed that half KQ conversion, not the score round-trip, accounts for the larger loss. |
| 20 | Restore a single vector 16-byte LDS load for each Q fragment instead of four scalar half2 LDS reads. Iteration 20a median 247.616 / 348.029 ms (1.405×), synthetic max error `7.16e-9`; 83 VGPR / 48 SGPR, zero spills/private bytes, 29,056 B LDS. ISA: 11 `ds_read_b128` vs 18 `ds_read_b32` (previously 3 vs 46), 32 `v_dot2_f32_f16`, 8 barriers, vector global loads, no scratch. | Isolated gate passed. |
| 20b | Move scale before Q→FP16 conversion as pinned mx does; same geometry and vector loads. Median 247.812 / 348.272 ms (1.405×), synthetic max error `7.16e-9`. | No material change in speed or real-output error. |

Iteration 20b's active M26 P8216 run measured 43,466.51 ms prefill and
8,039.74 ms decode (51,522.43 ms total), versus one matched control at
51,394.08 ms prefill and 8,268.80 ms decode (59,671.39 ms total). Do not claim
this as a qualified end-to-end win: the candidate/control order was not
thermally interleaved, and the candidate had the one-shot comparison sync. The
layer-3/base-512 output still differed by `max_abs=4.97699e-5`,
`rms=5.07132e-7`. Candidate and control matched for the first 36 generated
token IDs; the first divergence was token index 36. Moving the scale before Q
conversion changed neither mismatch magnitude nor divergence index. The
reference-shaped reuse principle is now spill-free and >15% faster in isolated
P8192 A/B, but greedy correctness remains the active blocker. Do not qualify
the full context curve or promote until that is resolved.

### Iteration 13 hypothesis / gates (archived)

Restore iteration 10's fast FP16-Q/in-place-FP32-weight kernel and change only
the suffix split count from 32 to the source-pinned P8192 value of 3. Keep the
`base_position == 0` control path untouched. This is reference-derived, not a
split sweep: both pinned reference traces use 3 splits. Existing 3-split
isolated P8192 data already clears the ≥15% gate. Correctness gate is exact
128-token greedy IDs against control; if it passes, repeat the request A/B and
record the end-to-end delta. If parity fails, reject this split contract and
resume root-cause analysis without a further split sweep.

### Iteration 12 hypothesis / gates (archived)

Use a two-term FP16 expansion per query value: `q_hi = half(q)` and
`q_lo = half(q - float(q_hi))`; accumulate `K·q_hi` and `K·q_lo` with the same
`v_dot2` path. This changes only Q representation/arithmetic relative to
iteration 10 and targets its measured real-input discrepancy. Dynamic LDS is
45,440 B; require zero spills/private bytes and record VGPR/SGPR before timing.
Correctness gate: exact 128-token greedy IDs plus the real layer-3/base-512
operand comparison. Performance gate: five interleaved 32-split isolated
P8192 pairs and ≥15% median win. Kill if parity fails, resources spill, or the
performance gate misses; do not run repeated full-model timing on a loser.

| Iteration | Resource / correctness / P8192 result | Disposition |
|---|---|---|
| Reference-shaped 16×2 KQ-fragment path, iteration 1 | Resource/ISA passed; P512/P2048/P4096/P8192 FP16 control max abs error `7.16e-9`, finite; P8192 was 2.33–2.37 s vs 348–350 ms control (0.147–0.150x speedup ratio) | revise, not promote |
| 2 — `{32,8}` warp mapping | Correct; P8192 1.179–1.181 s; 0.295–0.296× control/candidate | revise |
| 3 — native `v_dot2_f32_f16` with dimension-lane mapping | Correct; P8192 1.371–1.390 s; slower than iteration 2 | reject isolated change |
| 4–5 — precomputed KQ weights and row-reuse order | Correct within `3.84e-6` max abs; P8192 1.252–1.296 s | revise |
| 6 — reference K-position lane ownership, 64×256 K tile | Correct; P8192 625.4–625.8 ms; 0.557–0.619× control/candidate | revise |
| 7 — exact `nbatch_fa=32`, `nbatch_K=128`, `nbatch_V=16` | Resource/ISA passed; correct within `3.84e-6` max abs; P8192 466.1–466.4 ms vs 348.1–355.3 ms control; 0.747–0.762× | revise, not promote |
| 8 — vector 16-byte KQ fragments, tile sizes held fixed | Resource/ISA passed; P8192 median 245.861 ms vs 348.894 ms control (1.42× speedup; 29.5% less time) | promote to narrow e2e |
| 9 — FP32 Q and FP32 tile weights | Correct to `5.59e-9` max abs; P8192 median 400.021 ms vs 348.328 ms control (0.87×) | reject this representation; revise |
| 10 — FP16 Q, FP32 weights in-place in KQ tile | 77 VGPR / 48 SGPR, zero spills, 29,056 B LDS; P8192 median 244.307/347.721 ms at 3 splits and 266.254/339.041 ms at production 32 splits (candidate/control; 1.273×) | exact 16-token parity passed after matching production route; continue model-level qualification |
| 11 — preserve FP32 Q, change QK arithmetic only | 86 VGPR / 48 SGPR, zero spills, 45,440 B LDS; real output error fell 94× but 16-token text still differed; P8192 468.901/338.587 ms at 32 splits | reject: 27.8% slower and still incorrect |
| 12 — FP16 high + residual query expansion, `v_dot2` | 77 VGPR / 48 SGPR, zero spills, 45,440 B LDS; synthetic max abs `3.73e-9`; P8192 895.534/338.445 ms at 32 splits | reject: 62.2% slower |
| 13 — iteration 10 precision, reference-pinned 3 splits, preserve base-0 control | 128-token output diverged at token 6 (candidate ID `7701`, control `5686`) | reject: exact greedy parity failed |
| 14 — per-query split boundaries, 3 splits, preserve base-0 control | 77 VGPR / 48 SGPR, zero spills, 29,056 B LDS; isolated P8192 median 250.227/348.829 ms (candidate/control, 1.394×); active M26 P8216 pair diverged at token 36, with 43.53/8.03 s prefill/decode vs 51.97/10.20 s control; real layer-3 attention max error `4.98e-5` | learn: fast but not exact; do not promote |
| 15 — FP32 Q, corrected per-query split semantics | 83 VGPR / 48 SGPR, zero spills/private bytes, 45,440 B LDS; synthetic error `3.73e-9`; P8192 median 644.709/349.106 ms (candidate/control, 0.541×) | reject: accuracy path misses isolated performance gate |

### Iteration 15 hypothesis / gates — FP32 query staging

Reopen the previously rejected FP32-Q representation only because iteration 14
fixed the per-query split intervals that had confounded its earlier model-level
correctness check, and the newly active M26 route now has direct real-input
evidence: iteration 14 diverged at greedy token 36 and showed `4.98e-5`
maximum attention-output error at layer 3/base 512. Change only Q staging and
QK arithmetic to FP32; keep the 16×2 row ownership, K/V tiles, three splits,
per-query split boundaries, KV precision, and reduction unchanged.

* Resource: 256 threads; zero VGPR/SGPR spills and private bytes; LDS at or
  below the prior 45,440-byte FP32-Q footprint. Verify the newly compiled
  gfx906 object before timing; expected occupancy ceiling is one CTA/CU by LDS.
* Correctness: real layer-3/base-512 output error must materially shrink, then
  exact 128-token greedy IDs must match on the active M26 P8216 request.
* Performance: five interleaved isolated P8192 pairs; require ≥15% median
  attention win. Only then run the active-route model A/B.
* Kill on resource failure, no material real-output error reduction, exact
  token mismatch, or failure of the isolated performance threshold. Prior
  I11's isolated slowdown is not waived; this corrected-boundary rerun must
  clear the same gate on the active launch path.

The reference trace and iteration-14 A/B establish the mechanism and target;
the max end-to-end ceiling remains bounded by the measured 12.86% attention
device-work share. Do not test another split count or tile geometry here.

### Reference configuration correction (2026-09-28)

Re-reading the pinned AMD config table directly corrected the earlier
`nbatch_fa=64` note: `GGML_CUDA_FATTN_TILE_CONFIG_CASE(256,256,16,256,2,32,128)`
selects `nbatch_fa=32` and `nbatch_K=128` for `ncols=32`. The V fragment is
`nbatch_V=16`. This also explains mx's recorded 27,136 B LDS footprint. The
first six candidates used an oversized 64×256 K tile; iteration 7 now uses the
source-selected tile/chunk sizes. The earlier EXP-V2-0042 dispatch table should
be read with this correction; P8192 CTA/query geometry and measured request
attribution are unchanged.

Iteration 8 changed only the KQ fragment load from scalar half2 LDS reads to
16-byte vector reads, retaining the reference-selected tile sizes. This is the
first candidate to clear the promotion gate. The next step is an opt-in,
P8192-only end-to-end A/B through the production wide-prefill call site; the
ordinary runtime remains unchanged. This isolated synthetic correctness result
does not yet establish model-level/token parity.

### Iteration 8–10 evidence (2026-09-28)

The opt-in microbenchmark compares FP16 split attention plus its existing
stage 2 against this candidate with the same split count. `--v2-0043` uses the
historical 3-split geometry; `--v2-0043-32split` matches the active PrefillV2
suffix route. Both paths run only four correctness points and five
interleaved P8192 A/B pairs; neither invokes the legacy full curve.

* Iteration 8 used 256 threads (`32×8`), 31,104 B dynamic LDS, 81 VGPR,
  43 SGPR, zero spills/private bytes. LDS permits at most two resident CTAs/CU
  by resource arithmetic. KQ uses explicit 16-byte reads and `v_dot2`.
  Synthetic max absolute error was `3.83891e-6`; P8192 median was 245.861 ms
  vs 348.894 ms control. It changed only the KQ load to the pinned `cpy_nb`
  vector-fragment mechanism and passed the isolated ≥15% gate.
* Iteration 9 used FP32 Q and FP32 weights. It matched the FP16 control to
  `5.58794e-9` max absolute error, but median P8192 time regressed to
  400.021 ms vs 348.328 ms control. This representation is rejected.
* Iteration 10 retained FP16 Q / `v_dot2`, but stores FP32 softmax weights
  in-place over the consumed KQ scores, eliminating the separate weight tile.
  It uses 29,056 B LDS, 77 VGPR, 48 SGPR, zero spills and private bytes; at
  most two CTAs/CU by resource arithmetic. Synthetic max abs error is
  `7.15954e-9`; five P8192 pairs had medians 244.307/347.721 ms (candidate /
  control) at 3 splits. The production-aligned 32-split run measured medians
  266.254/339.041 ms (candidate/control), a 1.273× isolated speedup; all five
  pairs favored the candidate and 512/2048/4096/8192 correctness checks were
  finite with the same max absolute error. The initial e2e candidate run changed both the first-chunk attention
  algorithm and the later suffix algorithm: `PrefillV2Model` processes 512-token
  macro tiles; for `base_position == 0`, control selects tiled-online attention,
  while the candidate had forced the 32-split path. Therefore the observed
  greedy mismatch and 39.117 s vs 40.014 s wall times are confounded and are
  not a valid kernel parity/performance verdict. The selector now activates
  only for `base_position > 0`, preserving the first-chunk control route. With
  that correction, a matched P8192 request generated identical 16 token IDs:
  `248068,271,248069,271,2064,5686,1040,488,2908,3162,290,264,1546,1248,886,314`.
  One end-to-end pair measured 39,278.6 ms candidate and 39,971.6 ms control;
  this is not enough repetition to claim a win. A 128-token greedy A/B then
  matched exactly for the first 66 tokens but diverged near token 67. A same-
  operand comparison at layer 3/base 512 measured candidate/control attention
  max abs `8.38637e-5`, RMS error `1.25813e-6`, with actual query max abs
  `11.6942`, RMS `1.19865`, and FP16-rounding max abs `0.0039053`. The isolated
  32-split gate passes, but exact greedy parity does not. Iteration 11 reduced
  real attention error 94× but failed parity and speed; iteration 12's
  compensated query was much slower; iteration 13's source-pinned 3 splits
  diverged at token 6. Iteration 14 now targets the measured control partition
  mismatch: the reference split interval is per query token, not per CTA tile.
### Iteration 21 — vectorized FP32-Q accuracy branch (2026-09-29)

Hypothesis: the earlier accurate FP32-Q variant's slowdown came from scalarized
LDS query loads; vectorizing those loads could retain accuracy while recovering
the P8192 win. The isolated build had 113 VGPR, 48 SGPR, zero spills/private
bytes, and about 44.4 KiB dynamic LDS (one CTA/CU by LDS resource arithmetic).
Synthetic correctness passed at 512/2048/4096/8192 with max absolute error
`3.72529e-9`. Five P8192 pairs had control/candidate medians 347.762/405.399
ms (0.858× control/candidate), failing the ≥1.15× gate. REJECT; do not route
this branch through the full model. The temporary FP32-Q kernel/API/benchmark
branch was removed, leaving only the FP16-Q candidate. Its isolated speedup
does not clear the independent real-model greedy-parity blocker (first
divergence at token 36).

After removing the FP32-Q branch, the remaining FP16-Q candidate was rebuilt
and rerun: synthetic max absolute error remained `7.15954e-9`; P8192 median
control/candidate was 347.717/247.707 ms (1.404×). This reconfirms the isolated
speed result, not real-model correctness or promotion.

### Current-build oracle and P8216 check (2026-09-29)

On the exact P8216 raw prompt (`"hello " × 8192` plus the recorded history-of-
computing instruction), greedy, 128-token runs produced this token-parity
matrix:

| Pair | First differing generated token | Result |
|---|---:|---|
| FP16-Q candidate vs MIInfer control | 36 | mismatch |
| MIInfer control vs pinned mx | 36 | mismatch |
| MIInfer control vs pinned upstream | 36 | mismatch |
| pinned mx vs pinned upstream | none | exact 128/128 IDs |
| FP16-Q candidate vs pinned mx | none | exact 128/128 IDs |
| FP16-Q candidate vs pinned upstream | none | exact 128/128 IDs |

The candidate and control sequences were compared with actual
`completion_probabilities[].id` arrays from the pinned mx server (commit
`6e4ef6c1`) and upstream server (build commit `73a43d1`), each run against the
same raw P8216 prompt with context 16,384, `n_predict=128`, `temperature=0`,
`top_k=1`, `top_p=1`, `repeat_penalty=1`, and `n_probs=1`. Both references
reported 8,216 evaluated prompt tokens and generated 128 tokens. The candidate
matches both references 128/128; the references match each other 128/128. The
MIInfer control differs from both references at generated index 36 (control
ID 781; reference/candidate ID 310). This resolves the prior apparent
reference split: the candidate's exact model-level sequence is the one
corroborated by both independent reference builds, while the control is the
outlier for this prompt.

The saved `/tmp/v2043-e19bb7a-upstream-parity.log` was mislabeled for this
purpose: its banner shows the MIInfer runtime and its IDs exactly match the
control, so it is not used as reference evidence. The direct same-input
layer-3/base-512 candidate/control comparison remains
`max_abs=4.97699e-5`, `rms=5.07132e-7`, with 20,173 exact outputs out of
3,145,728. This numerical difference explains that local disagreement but is
not being used to override the matching full-model result from both
references. The reference runs' wall times were sequential and not
interleaved; they establish token correctness only, not performance.

The unpaired end-to-end samples for that same prompt were candidate 51.450 s
(43.525 s prefill + 7.915 s decode), pinned mx 44.631 s (39.345 + 5.286), and
pinned upstream 51.118 s (45.137 + 5.981). The MIInfer sample includes the
real-compare synchronization; the three runs were sequential, not thermally
interleaved, and clocks/temperature were not captured together. Treat these as
directional only: candidate output now matches both references exactly, but
this candidate sample is slower than pinned mx and has not demonstrated the
required end-to-end win. Do not start the full curve. Next, measure the exact
P8192 competitive request repeatedly against pinned mx with debug comparison
sync disabled; the isolated P8192 attention gain and now-confirmed P8216 token
parity justify this narrow full-model gate.

### Re-evaluation — exact reference token capture (2026-09-29)

The ambiguous `upstream-parity.log` identity was resolved by rerunning the
exact P8216 request through both reference servers' `/completion` endpoints
with `n_probs=1` and comparing returned generated token IDs directly. Saved
responses are `/tmp/v2043-p8216-upstream-completion.json` and
`/tmp/v2043-p8216-mx-completion.json`. Both returned
`tokens_evaluated=8216`, `tokens_predicted=128`; their token IDs match each
other and the cleaned FP16-Q candidate 128/128. The saved MIInfer control IDs
still first differ at index 36. The earlier standalone CLI text-retokenization
attempt is superseded by these direct sampler-ID arrays and is not used as a
correctness result.

### P8192 exact-request end-to-end A/B (2026-09-29)

The material isolated P8192 gain (1.404x median) and exact P8216 match to both
reference engines justify the narrow full-request gate. MIInfer used the
saved request `/tmp/mi50-p8192-trace-v2-miinfer.2PPvwn/request.json`,
`m25_hi_qualified`, context 16,384, candidate enabled, session reuse disabled,
and the clean no-compare-sync route. The reference was the trace-matched
`mx-llama.cpp` server build `2e9d29f`, with prompt caching disabled and one
slot; it received the exact formatted 8,192-token prompt from the saved mx
trace response. All requests used greedy one-token generation. The first
candidate request generated ID `248068`; direct token capture on the mx raw
prompt also returned `248068`.

| Runtime | Client wall (s) | Internal prefill (s) | Prompt tokens | Reuse |
|---|---|---|---:|---|
| MIInfer candidate | 38.8600, 38.8474, 38.8535 | 38.8506, 38.8383, 38.8448 | 8,192 each | 0 session/prefix tokens |
| mx `2e9d29f` | 38.8925, 38.8992, 38.5879 | 38.8740, 38.8803, 38.5681 | 8,192 each | `cache_n=0` each |

Client-wall medians are 38.8535 s MIInfer and 38.8925 s mx, a nominal 39 ms
(0.10%) MIInfer advantage. That delta is far below the mx sample range
(311 ms); classify this result as **end-to-end parity within observed noise**,
not a performance win. SCLK/HBM were observed at 1606/1000 MHz; candidate
junction temperature peaked at 53 C. Per-sample mx temperature/power telemetry
was not retained, further limiting a sub-1% claim. A diagnostic rerun with
prompt caching enabled reused 8,188 tokens and completed in 0.23 s; that
sample is explicitly excluded. An earlier 6e4ef6c mx-build/chat-template run
at 55.97 s also used the wrong comparison binary/request path and is excluded.

This clears only the narrow P8192 no-regression check provisionally. It does
not promote the candidate: the full request has no demonstrated material
advantage, and the remaining long-context context-regression and refreshed
attention-family trace gates still need evidence. Continue with those
qualification checks; do not change query arithmetic based on this wall-time
tie.

### Re-evaluation — candidate route and shorter-context serving checks (2026-09-29)

The saved `/tmp/mi50-p8192-trace-v2-miinfer.2PPvwn/2746420_hcc_ops_trace.txt`
was captured Sep 28, before the candidate implementation commit
`7082b2b`, and contains only the old tiled/split-K control kernels. It is a
control trace and is not evidence of candidate P8192 kernel attribution. The
prior exact-request, no-compare P8192 timing remains a candidate-enabled
serving sample, but the source trace did not prove candidate dispatch.

Reran the exact saved 8,192-token request with
`MIINFER_V2_0043_GQA_ATTENTION=1` and diagnostic
`MIINFER_V2_0043_COMPARE_REAL=1`. Live logs confirm 15 candidate suffix calls
at base positions 512 through 7,680 in steps of 512, each with `token_count=512`.
The first real layer-3/base-512 candidate/control compare reported Q max abs
`11.6942`, candidate attention max abs `8.38935e-5`, RMS `1.25813e-6`, and
2,020 exact outputs of 3,145,728. The request completed 8,192 prompt tokens
and one generated token. Internal prefill was `38.9308 s`; exclude this
instrumented run from performance timing because the comparator synchronizes
and runs the control attention a second time. The older no-compare median
remains the only performance evidence. Candidate route coverage is now
proven; a fresh candidate attention-family trace follows.

The opt-in route correction was applied to both
`PrefillV2AttentionLayer::forward()` and `forward_profiled()`: all candidate
eligibility conditions (suffix position, no state object for the ordinary
path, FP16 KV, complete 16-token tile, and workspace) are now dispatch guards;
ineligible shapes retain their prior kernel. The target rebuilt and
`graphify update .` completed.

One-request-per-case serving samples used the same repeated-`hello` prompt
construction, greedy one-token generation, context capacity 16,384, and no
session/prefix reuse. Candidate was run before control on each context:

| Prompt tokens | Candidate prefill ms | Control prefill ms | Candidate/control | Observation |
|---:|---:|---:|---:|---|
| 2,048 | 8,962.85 | 9,061.93 | 0.989× | No apparent regression; one sequential pair only. |
| 4,096 | 18,380.3 | 18,719.1 | 0.982× | No apparent regression; one sequential pair only. |

Both returned HTTP 200, one completion token, and `finish_reason=stop`; token
IDs were not captured in these samples. With
`MIINFER_V2_0043_COMPARE_REAL=1`, the 2,048-token route logged candidate calls
at base positions 512, 1,024, and 1,536; the base-512 compare gave max abs
`2.01896e-4`, RMS `1.28693e-6`, max relative `6.3451`, and 11,288 exact
outputs of 3,145,728. This is real-input evidence and must not be conflated
with the `7.16e-9` synthetic/isolation result. The relative-error maximum is
near-zero sensitive; the RMS and token-level impact remain the useful checks.
These single, non-interleaved A/Bs are preliminary only; repeat them with
interleaving and captured hardware telemetry before claiming qualification.

The earlier 521-token request exercised a 512-token prefix plus a 9-token
tail, so the candidate correctly did not run there. Its HTTP 200 verifies the
fallback fix, not P512 candidate performance. P512 remains control-route
coverage only; the P512 isolated attention gate had already passed.

### Re-evaluation — fresh candidate request-window trace (2026-09-29)

The exact saved 8,192-token request was captured with ROCTracer and the
candidate enabled. Raw HCC operations are in
`/tmp/v2043-p8192-candidate-trace-hip-20260929/3594339_hcc_ops_trace.txt`;
the server log reports 8,192 prompt tokens, zero prefix/session reuse, one
generated token, and 38.9549 s internal prefill. Tracing perturbs wall time,
so use only the per-kernel activity durations below for attribution, not as an
end-to-end A/B sample.

| P8192 MIInfer kernel family | Control calls | Control time (ms) | Candidate calls | Candidate time (ms) |
|---|---:|---:|---:|---:|
| Suffix stage 1 / KQ-V | 240 `qwen35_splitk_suffix_attn_stage1_quant` | 5,034.363 | 240 `qwen35_kq_fragment_reuse_attention_batch_f16` | 4,110.671 |
| Split-K combine | 240 | 22.057 | 240 | 22.158 |
| Prefix tiled online attention | 16 | 42.860 | 16 | 42.805 |
| Attention-family total | 496 | 5,099.280 | 496 | 4,175.634 |

For the same trace sets, pinned mx commit `2e9d29f` recorded 304
`flash_attn_tile` calls totaling 1,743.247 ms and 304 combines totaling
42.444 ms (1,785.691 ms total). Upstream commit `73a43d1` recorded 304 main
calls / 1,743.682 ms and 304 combines / 42.453 ms (1,786.135 ms total); its
recapture includes the model's BOS, so it is one evaluated token longer than
the exact 8,192-token MIInfer/mx request. Thus candidate attention-family work
is still 2.34× pinned mx, a 2.390 s residual attention gap. The candidate
reduces MIInfer's prior attention-family time by 18.11%, but has not closed
the competitor gap; the reuse principle is present, while the machine-level
schedule still needs improvement.

Candidate suffix KQ/V work is 18.35% faster in the trace; the attention-family
total is lower by 923.646 ms (18.11%). Launch count remains 496: each candidate
main kernel replaces the old stage-1 kernel, while the separate combine is
preserved. The 240 candidate calls equal 15 suffix chunks × 16 GQA layers;
the 16 prefix calls remain on the control tiled-online kernel. Thus the
reference-shaped rewrite is demonstrably active in the exact P8192 request,
and the attention-work reduction is now directly measured rather than
extrapolated from an isolated benchmark. The reference trace still has a
larger attention-family advantage than MIInfer, so the end-to-end competitive
gate remains parity-within-noise rather than a robust win.

### Re-evaluation — repeated P2048/P4096 serving A/B (2026-09-29)

Each case uses the same greedy one-token repeated-`hello` prompt, context
capacity 16,384, and disabled session/prefix reuse. Three samples per mode
were collected in fresh candidate/control server processes; within each
mode, P2048/P4096 requests were alternated. Candidate always preceded
control, so this is repeated but not thermally interleaved, and no per-request
clock/temperature telemetry was captured. Treat as a no-regression check, not
a speed claim.

| Prompt tokens | Candidate samples (ms) | Candidate median/range (ms) | Control samples (ms) | Control median/range (ms) | Median delta |
|---:|---|---:|---|---:|---:|
| 2,048 | 8,962.85 / 8,968.78 / 8,969.46 | 8,968.78 / 6.61 | 9,061.93 / 9,068.04 / 9,069.72 | 9,068.04 / 7.79 | −99.26 ms (−1.09%) |
| 4,096 | 18,380.3 / 18,392.1 / 18,418.2 | 18,392.1 / 37.9 | 18,719.1 / 18,741.5 / 18,778.7 | 18,741.5 / 59.6 | −349.4 ms (−1.86%) |

All requests returned HTTP 200 with one generated token and
`finish_reason=stop`; token IDs were not captured. The candidate distributions
do not overlap the control distributions, but the fixed candidate-then-control
order and absent telemetry prevent a qualified improvement claim. They show
no unacceptable P2048/P4096 regression in this run. P512 remains an
ineligible-tail fallback, so the selector guard leaves its established route
unchanged rather than routing a sub-16-token tail through the candidate.

### P512 selector-route correction (2026-09-29)

The first P512 chat request returned HTTP 500 before GPU work because
`PrefillV2AttentionLayer::forward()` applied the opt-in kernel to a call with
live `DevicePrefillState`. The initial fix restricted dispatch to stateless
calls, but retry exposed another valid unsupported shape (KV/workspace or
tile eligibility) still throwing. Dispatch now enters the candidate only when
all conditions are met: suffix position, no prefill state, FP16 KV, at least
16 tokens in a full 16-token tile, and Split-K workspace. Otherwise the
established attention route runs unchanged. The `miinfer` target rebuilt and
`graphify update .` completed. A repeat one-token greedy chat request returned
HTTP 200 and completed 521 prompt tokens at 2,738.9 ms prefill; this confirms
the request no longer errors, but it is not an A/B timing and the request
followed fallback rather than proving candidate coverage. The original
500s, the transient orphan-process OOM during a subsequent launch, and this
successful fallback request do not count as candidate P512 qualification.

An intermediate hand-assembled active-route smoke omitted part of the recorded
M26 environment and crashed in `RecurrentLayer::prefill_wide` before reaching
attention; its 14.2-GiB systemd core is preserved at
`/var/lib/systemd/coredump/core.miinfer.1000.0da260cadc72442a8fdc8904728b1a57.3327013.1790639695000000.zst`.
That run is not candidate evidence. The sourced complete vector and
core-disabled rerun above reached the intended attention comparison.

The pinned reference source/build audit narrowed the token mismatch: both
builds are Release for gfx906 with the same HIP compiler path and `-O3`; the
selected `flash_attn_tile` arithmetic body in `fattn-tile.cuh` is unchanged.
The file-level differences are the newer upstream `use_sparse=false` launch
argument. `fattn-common.cuh` differs in Mx's OOM-adaptive parallel-block
fallback / override, which upstream removed, and in fully-masked-row handling;
the latter is unreachable for this nonempty causal request, and the captured
P8192 reference grid uses three splits. Sparse-mask handling is disabled here.
These source differences do not explain why the two reference executables
first diverge at token 61; candidate parity with upstream is strong evidence
for the reconstructed tile, but pinned-mx parity remains an open correctness
comparison rather than a reason to change Q precision speculatively.

### Exact trace-shaped P8192/TG1 check (2026-09-29)

The current `e19bb7a` candidate server was run with the complete recorded
`m25_hi_qualified` environment, context 16384, and the exact saved P8192/TG1
request body from `/tmp/mi50-p8192-trace-v2-miinfer.2PPvwn/request.json`.
The first request completed with token ID 248068 and `prefill_ms=38837.7`
(38.850 s wall). The earlier trace samples were MIInfer control 39.881 s,
pinned mx 39.038 s, and pinned upstream 44.956 s. This single candidate sample
is nominally 0.20 s faster than mx and 1.04 s faster than MIInfer control, but
does not establish a win because it is unpaired and within plausible run
variance.

An immediate same-process repeat was attempted. The MIInfer process aborted
with `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` before returning a response;
therefore no repeat timing exists and the first sample remains preliminary.
This exposes a repeat-request/runtime stability issue (or an as-yet-unisolated
candidate interaction) that must be resolved before interleaved qualification.
Do not promote the candidate or claim it beats the pinned mx trace baseline.
Next isolate whether the failure reproduces with the control attention path
under the same server/request lifecycle, then continue same-state A/B only
after repeat requests are stable.

The first control-path isolation was not configuration-matched: that server
had prefix/session reuse enabled, unlike the candidate server. Repeating the
control test with `MIINFER_SESSION_REUSE=0` completed both same-process
requests successfully at 40.049 s and 40.012 s prefill (40.069 s and 40.012 s
client wall). The earlier control crash therefore does not establish a
general repeated-prefill failure. The candidate was already logged with reuse
disabled and still aborted on its same-process repeat, while the no-reuse
control is stable. Treat the candidate's repeated-P8192 HSA aperture violation
as a candidate-path memory-safety blocker; its single-request 38.838 s sample
is not eligible for performance promotion until resolved. That sample is
nominally ~3.0% faster than the no-reuse control pair median (~40.03 s), but it
is unpaired and cannot outweigh the repeatability/correctness failure.

### Repeat-fault synchronization localization (2026-09-29)

Fresh no-reuse P8192/TG1 two-request diagnostics show the candidate fault is
synchronization-sensitive. The uninstrumented candidate repeat previously
aborted with an HSA memory-aperture violation; the no-reuse control completed
two requests. With either `AMD_SERIALIZE_KERNEL=3` alone or
`AMD_SERIALIZE_COPY=3` alone, the candidate completed both same-process P8192
requests. The existing real-output compare hook (which synchronizes after the
layer-3/base-512 candidate/control comparison) also completed two requests.
Finally, an opt-in `MIINFER_V2_0043_SYNC_FIRST=1` diagnostic that synchronizes
only after the first candidate attention call at layer 3/base 512 completed two
requests at 38.860 s and 38.875 s prefill. All of these serialized timings are
diagnostic only; do not use them as candidate performance qualification.

Moving that one diagnostic wait to immediately before (rather than after) the
same first candidate call also completed two requests: 39.098 s and 38.840 s
prefill. This does not identify which side of the dispatch needs ordering: both
calls wait the same per-thread default stream, and the code path passes the
same stream through the preceding block and candidate launcher.

These sync experiments showed that draining the queue can suppress the earlier
fault, but did not identify a missing dependency. The temporary one-sync source
hook has been removed. A fresh no-sync replay then completed three same-process
requests with prefix reuse disabled; a second fresh no-sync replay with the
original `MIINFER_DUMP_TOKENS=1` setting completed two more. Both dumped
requests generated token ID `248068`. Their prefill times were 38.897/38.909 s.
The earlier HSA aperture fault is therefore not currently reproducible and is
not an established candidate memory-safety defect; retain it as an anomalous
failure, not as a resolved root cause. The 27-GB AMD GPU core dump is an AMDGPU
ELF core, but system GNU GDB cannot decode its device registers and no ROCgdb
executable is installed. The active qualification blocker is still FP16-Q
real-model numerical/token parity; do not promote from repeat stability alone.

### Request-window intermediate-traffic audit (2026-09-29)

Re-read the pinned gfx906 HIP configuration rather than using the similarly
named RDNA table: in `ggml/src/ggml-cuda/fattn-tile.cuh`,
`ggml_cuda_fattn_tile_get_config_amd()` selects `<256,256,ncols=32>` as
256 threads, occupancy target 2, `nbatch_fa=32`, `nbatch_K=128`. The RDNA
table's `<256,256,ncols=32>` entry has `nbatch_K=128` too, while its 16-column
entry is not the MI50 route. For observed `<16,2>`, `ncols=32`; the active
gfx906 `<16,2>` entry is therefore 32 positions per K tile, processed in two
128-dimension KQ chunks. Do not infer the active config from NVIDIA or RDNA
table entries.

The exact request-window HCC sums are:

| Implementation | Main attention calls / time | Separate combine calls / time | Total |
|---|---:|---:|---:|
| MIInfer control | 240 / 5,034.363 ms | 240 / 22.057 ms | 5,099.280 ms (including 16 prefix tiles) |
| MIInfer KQ-fragment candidate | 240 / 4,110.671 ms | 240 / 22.158 ms | 4,175.634 ms (including 16 prefix tiles) |
| Pinned mx | 304 / 1,743.247 ms | 304 / 42.444 ms | 1,785.691 ms |
| Upstream | 304 / 1,743.682 ms | 304 / 42.453 ms | 1,786.135 ms |

The MIInfer candidate workspace is `splits * tokens * query_heads *
(head_dim + 2) * sizeof(float)`. At 3 splits, 512 query tokens, 24 heads,
and D=256 this is 38,043,648 bytes per call. Stage 1 writes and stage 2 reads
that workspace, or approximately 76.09 MB per call and 18.26 GB for 240 calls.
This is substantial traffic, but the measured separate combine is only
22.16 ms. The reference combine is slower (42.44 ms), while the candidate main
kernel is 2.36x the reference main time (4,110.67 / 1,743.25). Thus the
candidate's extra split-partial round-trip cannot explain the observed main
kernel gap; the remaining target is main-kernel execution efficiency. The
reference also incurs 64 more main/combine calls in this captured request, so
launch count does not account for its advantage.

This accounting narrows but does not yet attribute the main-kernel gap to a
specific machine-level cause. The candidate meets the data-lifetime intent
(KQ fragment and shared-workspace reuse) but is not yet architecturally or
performance qualified. Do not promote or start a parameter sweep from this
result. Next inspect the candidate's generated gfx906 ISA/resource metadata
against the selected reference code object, focusing on memory instructions,
barriers, vectorization and row/warp ownership; then test only a directly
supported correction at isolated P8192.

### Iteration 21 hypothesis — reference-width KQ weights

The pinned fast-FP16 implementation stores normalized tile-local KQ weights
as half and consumes them with FP16 V fragments; the current candidate stores
the same fragment as FP32. Test only changing that fragment's storage to
FP16, converting values back to FP32 when the existing FP32 output accumulator
consumes them. Keep geometry, split count, QK arithmetic, K/V staging,
online-softmax state, output accumulators and combine unchanged. Expected
effects are half the KQ LDS footprint/traffic and closer reference fragment
representation, without introducing lower-precision output accumulation.

Baseline: current candidate source and latest request-window trace above
(4,110.671 ms candidate main-kernel work; 2.36x pinned mx). Gates: compile
256-thread gfx906 code with zero VGPR/SGPR spills and zero private bytes;
isolated control comparison at P512/P2048/P4096/P8192 remains finite and
within the already-recorded numerical envelope; five paired P8192 timings
must not regress versus the current candidate. Do not perform full-model
qualification unless the isolated P8192 result is both correct and measurably
faster. The attention-only Amdahl ceiling remains 12.86% of total device work.

Iteration 21 result: gfx906 build passed. Synthetic comparisons were finite at
P512/P2048/P4096/P8192, all with max absolute error `2.59047e-6` (relative
error peaks at `1.01262` only near zero-valued references, so absolute error
is the meaningful recorded comparison). Five P8192 pairs measured medians
347.879 ms control / 249.195 ms candidate; candidate ranged 249.065–293.589 ms
with one timing outlier. This remains a 28.3% isolated win over control but is
not faster than iteration 20's 247.616 ms median beyond likely run noise.
Decision: KEEP the FP16 KQ fragment as the current reference-shaped baseline;
LEARN that fragment width alone does not explain the residual gap. Continue
with a separate test of the reference's packed FP16 per-CTA V×KQ accumulator,
converting to FP32 only at split-partial writeout.

### Iteration 22 hypothesis — packed FP16 online output state

The pinned fast-FP16 path carries `VKQ` as `half2` through online max
rescaling and V×KQ accumulation, then converts to FP32 when writing split
partials. The current candidate keeps these per-CTA accumulators in FP32.
Change only the per-CTA output-state representation/arithmetic to packed FP16;
retain the FP16 KQ fragment, FP32 max/sum, FP32 split workspace/combine,
geometry, split count and load schedule. This tests the reference's vector
arithmetic/state lifetime directly and may reduce registers plus arithmetic
cost. Gate on finite P512/P2048/P4096/P8192 outputs and max absolute error
within the existing `2.7563e-5` attention envelope, zero compiler spills/private
bytes, and five paired P8192 timings faster than iteration 21's 249.195 ms
candidate median and no slower than control. Do not run full-model testing if
either correctness or isolated performance fails.

Iteration 22 result: P512/P2048/P4096/P8192 comparisons were finite with max
absolute error `1.15326e-5`, inside the existing attention envelope. Two five-
pair runs had candidate medians 241.062 ms and 241.065 ms; in each, four
candidate samples clustered at 240.958–241.307 ms and one was a ~281 ms
outlier. The paired controls were stable around 348 ms. Relative to iteration
21's 249.195 ms candidate median, the repeated ~241.06 ms result is a 3.3%
isolated improvement; relative to control it is a 1.44x speedup. The rebuilt
gfx906 code object reports 70 VGPR, 46 SGPR, zero VGPR/SGPR spills, and zero
private bytes. Dynamic LDS is 27,008 B (below the pinned object's 27,136 B);
the launch remains 256 threads. Keep this as the current candidate and proceed
to full-model greedy/token parity and the exact P8192 competitive request. The
timing outlier is retained, not dropped; report medians and ranges.

Greedy full-model parity at the exact raw P8216 prompt passed: iteration 22
generated 128/128 identical token IDs to both pinned mx and upstream reference
runs. The direct MIInfer CLI run measured 174.40 prompt tokens/s and 6.20
decode tokens/s (161.227 ms/token). This establishes parity, not competitive
latency; compare the exact P8192 request end-to-end against the saved pinned
reference request before any promotion decision.

### Iteration 22 — exact P8192 end-to-end recheck

The saved P8192/TG1 greedy request was run three times on the current iteration
22 candidate with `m25_hi_qualified`, context 16,384, and session reuse off.
Every run processed 8,192 prompt tokens and generated ID `248068`. Candidate
internal prefill times were 38,849.8 / 38,744.6 / 38,733.9 ms (median
38,744.6 ms; range 115.9 ms). The saved pinned-mx no-cache samples were
38,873.986 / 38,880.321 / 38,568.125 ms (median 38,873.986 ms; range
312.196 ms). The candidate median is 129.4 ms (0.33%) lower, but the ranges
overlap substantially and the pinned samples predate this rerun. Treat this as
end-to-end parity within observed noise, not a material win. Together with the
P8216/128-token exact greedy parity above, correctness now passes; competitive
performance and attention-family attribution for iteration 22 remain open.

### Re-evaluation — fresh iteration 22 trace and ISA audit (2026-09-29)

Captured one exact saved P8192/TG1 request on iteration 22 with the host
ROCtracer plugin, `m25_hi_qualified`, and session reuse disabled. It evaluated
8,192 prompt tokens and returned token ID `248068`. HCC operations are at
`/tmp/v2043-iter22-current-candidate-trace/3680137_hcc_ops_trace.txt`; HIP API
calls are at the adjacent `3680137_hip_api_trace.txt`. The trace contains
exactly 240 reference-shaped suffix calls, 240 split combines, and 16 prefix
online-attention calls, covering the request window.

| Iteration 22 family | Calls | HCC time |
|---|---:|---:|
| Candidate suffix KQ/V | 240 | 3,989.611 ms |
| Split combine | 240 | 21.927 ms |
| Prefix tiled online attention | 16 | 42.794 ms |
| Attention-family total | 496 | 4,054.332 ms |

The same pinned-mx trace records 304 `flash_attn_tile` calls / 1,743.247 ms
and 304 combines / 42.444 ms (1,785.691 ms total); upstream records
1,743.682 ms main / 42.453 ms combine (1,786.135 ms total). Iteration 22 is
2.270x the pinned-mx attention-family time; the main suffix kernel is 2.289x
the reference main-kernel time. Compared with the earlier candidate trace,
iteration 22 reduced main time by 2.95% but leaves a 2.269 s attention-family
gap. Its traced request prefill was 38.757 s; the attention gap therefore has
a 5.86% wall-time upper bound, large enough to remain material to the
competitive/closure target.

The extracted iteration-22 gfx906 candidate object reports 256 threads, 70
VGPR, 46 SGPR, zero spills/private bytes, and 27,008 B LDS. The selected pinned
mx object is 256 threads, 97 VGPR, 46 SGPR, zero spills/private bytes, and
27,136 B LDS. Static disassembly counts include 32 `v_dot2_f32_f16` sites in
the MIInfer KQ loop versus 512 in the pinned specialization. Source inspection
explains the code-shape difference: mx compile-time-unrolls its 16-element KQ
fragment loop; MIInfer's equivalent `fragment < 16` loop has no unroll
directive. The two loops do the same per-fragment dot work, but MIInfer carries
it through a compact loop body. This is direct compiler evidence for testing
KQ-loop scheduling, not proof that unrolling improves runtime. Both paths
remain LDS-limited to at most two resident CTAs/CU by resource arithmetic;
achieved occupancy was not measured. The candidate has 10 static barrier sites
versus 9 in mx and 99 static `s_waitcnt` sites versus 232; do not interpret
static counts as dynamic executions.

### Iteration 23 hypothesis — unroll only the KQ fragment loop

Change only the candidate's 16-iteration KQ `fragment` loop to compile-time
unrolled code, matching the pinned source's unroll policy. Keep geometry, KV
tile size, split count, FP16 KQ storage/output state, reductions, V path,
barriers, and launch contract unchanged. The isolated baseline is iteration
22 (P8192 median 241.06 ms; current control was approximately 348 ms). The
resource gate is zero VGPR/SGPR spills and zero private bytes, 256 threads,
and LDS no larger than the gfx906 workgroup budget. The correctness gate is
finite P512/P2048/P4096/P8192 outputs within the existing `2.7563e-5`
absolute-error envelope. The performance gate is five paired P8192 isolated
runs with at least a 5% median improvement over iteration 22 and no loss
against control. Do not run full-model qualification if code spills, numerical
correctness fails, or the repeated isolated win misses that threshold. If the
gate passes, repeat exact greedy parity and P8192 end-to-end timing; otherwise
retain iteration 22 and use the trace/codegen evidence to select the next
single reference-schedule divergence.
