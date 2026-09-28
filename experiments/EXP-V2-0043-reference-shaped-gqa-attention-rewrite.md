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

The direct full-model greedy check used `MIINFER_CONTEXT_CAPACITY=16384`,
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

The serving API cannot currently exercise the candidate: its batched caller
passes reusable prefill state and the opt-in path deliberately rejects that
mode before launching the kernel. The direct CLI path did exercise it without
that state. Keep the candidate opt-in and do not promote it based on isolated
attention timing alone. The structural reuse prototype passes exact 128-token
correctness and the isolated P8192 gate, but the current full-model check does
not show a speedup; an API-compatible, clock/thermal-controlled repeat is
still needed before promotion.

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
| 14 — per-query split boundaries, 3 splits, preserve base-0 control | 77 VGPR / 48 SGPR, zero spills, 29,056 B LDS; isolated P8192 median 250.227/348.829 ms (candidate/control, 1.394×); exact 128-token greedy IDs match | retain opt-in prototype; no promotion: one full-model pair was neutral (+0.02% prefill time) and serving API path rejects reusable prefill state |

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
