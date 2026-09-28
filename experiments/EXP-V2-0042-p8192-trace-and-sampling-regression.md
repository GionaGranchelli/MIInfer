# EXP-V2-0042 — P8192 trace attribution and sampler-change decode A/B

**Status:** `LEARN — TRACE AGGREGATION COMPLETE; REF DISPATCH GEOMETRY UNRESOLVED; SAMPLER CHANGE NOT THE ~20 MS/TOKEN GAP`

## Objective

Close the request-window/kernel-family attribution for the three existing
P8192 traces and test whether `3f412d9b` caused the decode loss by comparing
its serving path with its immediate parent `4eaa016b`. No new trace was
captured for P8192.

## P8192 trace inputs and method

The original HCC and HIP API traces were usable and retained under `/tmp`:

| Runtime | Capture directory | Request window |
|---|---|---|
| MIInfer | `/tmp/mi50-p8192-trace-v2-miinfer.2PPvwn` | `hardware.txt` `request_start/end` |
| mx-llama.cpp | `/tmp/mi50-p8192-trace-v2-mx.fyWitM` | `hardware.txt` `request_start/end` |
| upstream llama.cpp | `/tmp/mi50-p8192-trace-v2-upstream.jlbkwf` | `hardware.txt` `request_start/end` |

HCC monotonic event timestamps were mapped to the HTTP request's realtime
start/end using the same-boot realtime/monotonic offset. Only non-marker
kernel events fully inside each request interval were aggregated. The exact
request-window sums below are GPU device-work sums, not critical-path wall
time; overlapping launches/streams prevent interpreting their difference as
an end-to-end latency delta.

| Runtime | In-window kernels | Device-work sum | Outside-window kernels | Request wall |
|---|---:|---:|---:|---:|
| MIInfer | 20,899 | 39,645.183 ms | 1,043 | 39.903 s |
| mx | 35,084 | 38,552.376 ms | 5,284 | 39.038 s |
| upstream | 35,228 | 44,481.246 ms | 5,172 | 44.956 s |

Kernel-family aggregation, in milliseconds:

| Family | MIInfer | mx | upstream |
|---|---:|---:|---:|
| Q4 projection/MMQ + quantization | 30,717.859 | 32,744.261 | 36,900.128 |
| Attention (all stages) | 5,099.280 | 1,784.926 | 1,785.428 |
| GDN | 2,029.618 | 1,521.172 | 3,254.079 |
| Other | 1,798.426 | 2,502.017 | 2,541.611 |

Dominant symbols were:

| Runtime | Kernel | Calls | Sum (ms) |
|---|---|---:|---:|
| MIInfer | `mx_repacked_mmq_legacy_kernel` | 6,400 | 30,318.377 |
| MIInfer | `qwen35_splitk_suffix_attn_stage1_quant_kernel` | 240 | 5,034.363 |
| MIInfer | `qwen35_splitk_suffix_attn_stage2_kernel` | 240 | 22.057 |
| MIInfer | `mx_gdn_chunk_kernel` | 768 | 2,029.618 |
| mx | `mmq_gemm_repacked` | 6,400 | 32,459.727 |
| mx | `flash_attn_tile` | 272 | 1,742.605 |
| mx | `flash_attn_combine_results` | 272 | 42.321 |
| mx | `gated_delta_net_chunked_cuda` | 768 | 1,521.172 |
| upstream | `mul_mat_q` | 6,400 | 36,617.021 |
| upstream | `flash_attn_tile` | 272 | 1,743.099 |
| upstream | `flash_attn_combine_results` | 272 | 42.329 |
| upstream | `gated_delta_net_cuda` | 816 | 3,254.079 |

Relative to mx, MIInfer's device-work delta is +3,314.354 ms attention,
+508.446 ms GDN, -2,026.402 ms projection/quantization, and -703.591 ms other,
for +1,092.807 ms net device work. The corresponding traced request-wall delta
is only +0.865 s; these sums do not define the critical path. Existing
untraced qualification had a 1.296 s median MIInfer-vs-mx gap (3.36%).

### Corrected selected reference schedule

The original HIP API trace records the reference symbol
`flash_attn_tile<256,256,2,2,false>`, with grid
`numBlocks={z=12,y=4,x=1}` and block `dimBlocks={z=1,y=8,x=32}`. This is
256 threads (four Wave64 waves), two Q positions and two Q-head slots per CTA,
with four K/V split partitions. The following
`flash_attn_combine_results<256>` launch has grid
`{z=1,y=24,x=2}` and 256 threads, combining split results for two query
positions and 24 query heads. Both pinned mx and upstream traces select this
same schedule. The earlier 4-Q-position × 8-Q-head-slot note described an
available code object, not the P8192 runtime dispatch.

MIInfer's trace selects
`qwen35_splitk_suffix_attn_stage1_quant_kernel<false,false,4>` with grid
`{z=3,y=512,x=12}` and 64 threads: one query position and two Q-heads per CTA,
three KV partitions, then a distinct stage-2 reduction over 12,288 CTAs.
This is the first concrete reference-derived reuse hypothesis to investigate;
it is not yet proof that reproducing the reference schedule will win.

Static code-object metadata for mx's selected reference tile is 20,000 bytes
LDS, 94 VGPR, 44 SGPR, Wave64, 256-thread maximum, and zero reported spills.
The old 27,136-byte/92-VGPR resource note was for the non-selected 4×8 code
object. Runtime occupancy and equivalent upstream metadata remain to be
confirmed during source-path analysis; do not infer achieved occupancy from
the static limits alone.

### Dispatch geometry validity caveat

The saved mx server log confirms an 8,192-token prompt, but the saved API trace
shows `grid.x=1` for every selected `<2,2>` tile launch. In the pinned source,
`grid.x=ceil(Q->ne[1]/ncols1)` and `ncols1=2` here, so that geometry represents
only 2–4 query positions per launch, not the 512-token microbatches expected
for this request. The 272 tile/combine launches also do not reconcile with
that query coverage. Thus the saved family-time aggregation remains useful,
but its launch dimensions cannot yet support an exact P8192 CTA/occupancy or
intermediate-traffic comparison.

A single exact-8192-token mx request was subsequently profiled with the pinned
binary through `rocprofv3` to resolve this contradiction. The request itself
completed (8,192 prompt tokens; one generated token), but the profiler wrapper
did not finalize its output on interruption and had to be stopped; its output
directory is empty. This targeted capture is therefore unusable and supplies
no replacement launch geometry. No MIInfer or upstream trace was rerun. Before
any P8192 attention prototype, resolve the mx dispatch dimensions using a
profiler run that exits normally or another trustworthy dispatch-level source.

## Decode regression A/B

### Setup

* Parent: `4eaa016bb6334df7b9639ac62dc55c34e9771745`.
* Candidate: `3f412d9bf3eec691f8eec664cba783cc563de44f` (immediate child).
* Each revision was built from a detached clean worktree with Release HIP,
  target `gfx906`.
* Same MI50, Qwen3.8-27B-Q4_K_M model
  `/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf`, same server route,
  16,384-token context, one resident request, no session/prefix reuse, and
  128 generated tokens.
* The P64 fixture was the established repeated-astronomy text; P2048 was 32
  copies separated by one ASCII space. The OpenAI chat wrapper adds eight
  tokens: reported prompt counts were 72 and 2,056 on both revisions.
* Requests explicitly used `temperature=0`, `top_p=1`, `top_k=0`, and
  `repetition_penalty=1` to match legacy greedy semantics. Completion count was
  128 for every sample. Candidate output SHA-256 matched the parent exactly:
  P64 `698b33c174ea934e32e1b1979326fc41e2ba5e76f3b99bb9da463d374e248c4d`;
  P2048 `bf3698237c0a156f4d089e3cf99f21772f7b2d1d7357d9a8cb827fdaea466c8b`.

### Results

Server-reported `decode_ms` covers all 128 generated tokens. The table gives
the two matched neutral-sampling samples per cell and their median divided by
128. Timing is per-token decode, not prefill-inclusive request time.

| Context | Parent samples (ms) | Parent median ms/token | `3f412d9` samples (ms) | Candidate median ms/token | Delta |
|---:|---:|---:|---:|---:|---:|
| P64 | 4,311.91; 4,329.01 | 33.75 | 4,373.92; 4,356.92 | 34.11 | +0.35 ms/token (+1.04%) |
| P2048 | 4,563.85; 4,568.84 | 35.67 | 4,601.46; 4,600.54 | 35.95 | +0.27 ms/token (+0.76%) |

This small directional slowdown is nowhere near the approximately 20 ms/token
competitive gap and does not attribute that gap to `3f412d9`. The result is a
narrow A/B (two paired samples per case, sequential server restarts), not a
full qualification: no direct clock/power telemetry was recorded, and the
sub-1.1% deltas are below what these samples establish as significant.

### Source attribution and required follow-up

The `3f412d9` change removes device-side argmax/state advancement from the
reusable decode graph and introduces host sampling in
`src/prefill_v2/model.cpp`: full-vocabulary logits are copied D2H, the host
synchronizes and applies sampler logic, then the next decode state is copied
H2D. That exact path is present in the tested serving engine. It is not the
cause of the measured large decode gap in this A/B.

However, temperature-zero requests still take the host-logits path in this
revision, contrary to the device-resident greedy requirement. Restore the
device-resident greedy/temperature-zero path. Non-greedy sampling also needs a
device-side candidate/reduction design; do not copy full-vocabulary logits to
CPU or rebuild decode state on host for every token. This behavior correction
is still required even though this A/B found no material timing regression.

## Decision / next work

`LEARN`: request-window/kernel-family aggregation is complete from the three
original captures. The reference source's `<2,2>` code object offers a
concrete query/head reuse principle, but the saved P8192 launch geometry is
contradictory and the targeted replacement capture did not produce usable
data. Keep exact runtime geometry, achieved occupancy, and intermediate
traffic open; do not begin a P8192 prototype until those are resolved. No full
curve is authorized for an unproven candidate.

The sampling A/B does not authorize further decode tuning or competitor decode
profiling. First fix the required device-resident greedy fast path; design
non-greedy candidate sampling without full-logits D2H and per-token host state
rebuild. Keep that work separate from the P8192 attention hypothesis.

### Pinned source-path audit (follow-up)

Audited the exact checkout revisions used by the original trace:

* mx: `2e9d29fe736969160f17476ecf0a6298cee6966`
* upstream: `73a43d1f69345aee8bb186ef4b3172cef892f2e5`

Both select `flash_attn_tile<256,256,2,2,false>` and the corresponding
`flash_attn_combine_results<256>` path for gfx906. In
`ggml/src/ggml-cuda/fattn-tile.cuh`, `<2,2>` means two query positions
(`ncols1`) and two query-head slots (`ncols2`) per CTA. GGUF metadata gives
24 Q heads / 4 KV heads (GQA ratio 6), so `grid.z=12` matches
`ceil(6/2) * 4`: the tile groups two of the six Q heads sharing a KV head.
The K/V tile work is reused across those two query positions and two head
slots. The gfx906 config
for `DKQ=DV=256,ncols=4` requests 256 threads, occupancy 2, `nbatch_fa=64`,
`nbatch_K=128`. Each CTA walks 64 KQ rows per online-softmax iteration, with
up to 128 K columns processed in parallel. The trace reports four KV
partitions (`grid.y=4`), then a separate combine launch. Selected mx
code-object metadata is 20,000 B LDS, 94 VGPR, Wave64, zero spills;
launch-bounds request two resident CTAs, but achieved/runtime occupancy was
not captured. Upstream uses the same selected tile schedule; its sparse-mask
support is CUDA-only and not active on this gfx906 path.

The stage-1 partial output and metadata are global scratch, reduced by the
separate combine kernel. At four splits this is approximately 9,280 bytes of
partial write + partial read + final write per query-head output (256 FP32
values plus `float2` metadata per split); total P8192 traffic cannot be
computed until true query coverage is resolved. MIInfer's stage 1 maps one
query token and a pair of Q heads per 64-thread CTA, with three KV splits;
stage 2 reduces global FP32 max/sum/accumulator partials. The reference reuse
principle is a plausible explanation for the measured attention-family delta
(1.785 s vs 5.099 s), but the trace/source mapping remains internally
inconsistent: API launches show grid `{x=1,y=4,z=12}` for `<2,2>`, which
covers only two Q positions along x, while the request was 8,192 tokens and
only 272 tile launches were attributed to it. Do not prototype until the
executed binary's configuration and query coverage are reconciled.

### Exploratory serving comparison (not qualification)

After the parent-vs-candidate sampling A/B, ran five alternating P64/TG128
and P2048/TG128 requests against current MIInfer and each pinned reference
server, with 128 generated tokens, explicit greedy-neutral settings, exact
input counts (72 / 2,056 including chat wrapper), and prompt-cache reuse off.
Reference JSON timing artifacts are in
`/tmp/mi50-serving-qualification.HF7aqD/` (not durable).

| Runtime | P64 median decode | P2048 median decode |
|---|---:|---:|
| MIInfer | about 34.0 ms/token | about 35.9 ms/token |
| mx pinned | 39.339 ms/token | 39.997 ms/token |
| upstream pinned | 44.712 ms/token | 45.309 ms/token |

MIInfer's per-request values are only recoverable from interactive daemon
output, not a saved server log, so they are approximate and the table is
diagnostic only. Telemetry covered the entire multi-runtime window rather than
labeled per-run windows: SCLK/MCLK stayed at 1606/1000 MHz, but sampled socket
power peaked at 250 W against the configured 225 W cap and junction
temperature reached 91 C. All engines produced 128 tokens with matching
prompt-token counts, but generated text diverged across engines, so this is
not a correctness comparison. Do not update the scoreboard or claim a
competitor win. A future decode comparison needs durable per-request timing
capture and timestamped telemetry labeling; no full curve is warranted.

### Durable narrow serving requalification — 2026-09-28

Repeated the narrow comparison with one daemon log per runtime, one telemetry
file per runtime, and raw request/response JSON retained under
`/tmp/mi50-decode-qualification.H5ei3f/`. This supersedes the diagnostic table
above for latency only; it does not supersede its warning about cross-engine
output divergence.

The protocol was five alternating P64/TG128 and P2048/TG128 measured requests
per runtime, after one warmup per prompt length; cache reuse disabled, exact
prompt counts 72 and 2,056 (including wrapper), 128 generated tokens, T=0,
top-p=1, top-k=0, repeat/presence/frequency penalties neutral. Each response
reported the expected prompt and generated counts. MIInfer server `decode_ms`
was recorded for every request; reference timings use `timings.predicted_ms`.
The table reports the median of five request durations divided by 128.

| Runtime (pinned build) | P64 ms/token (five raw ms) | P2048 ms/token (five raw ms) |
|---|---:|---:|
| MIInfer (`3f412d9b`) | 34.032 (4378.39, 4365.97, 4355.32, 4352.69, 4356.08) | 35.953 (4606.74, 4601.81, 4601.92, 4600.75, 4602.29) |
| mx-llama.cpp (`2e9d29fe`) | 39.5706 (5057.706, 5062.971, 5065.031, 5068.097, 5088.643) | 40.1920 (5140.844, 5143.350, 5144.579, 5147.670, 5151.887) |
| upstream llama.cpp (`73a43d1f`) | 44.4510 (5682.805, 5687.222, 5689.726, 5689.952, 5692.030) | 45.1015 (5765.638, 5770.849, 5772.998, 5779.487, 5786.901) |

Thus the current binaries do not reproduce a ~20 ms/token competitive decode
loss: MIInfer is 5.539 / 4.239 ms/token faster than pinned mx at P64/P2048 and
10.419 / 9.149 ms/token faster than upstream. These are serving-path numbers,
not a correctness or quality win. MIInfer's output-ID hash was stable across
its five repeats at each length, but generated text differed across engines;
the benchmark therefore cannot certify equivalent token sequences.

Environment: MI50 gfx906; model
`/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf`; ROCm clocks sampled
at SCLK/MCLK 1606/1000 MHz for all three windows. The configured sysfs PPT cap
was 225 W (`power1_cap`), equal to the reported hardware maximum. Sampled
socket-power peaks were 246 W MIInfer, 248 W mx, and 258 W upstream; hotspot
maxima were 88 C, 84 C, and 90 C respectively. The AMD telemetry API documents
that socket-power samples can rarely exceed the limit, so these observations
alone do not establish a different cap or invalidate one runtime selectively.
They do mean instantaneous cap adherence is not proven; all engines were run
under the same configured cap and clock state. No cap or clock setting was
changed for this comparison.

The earlier parent-vs-candidate T=0 A/B (two samples per cell) remained nearly
neutral: P64 33.75 to 34.11 ms/token and P2048 35.67 to 35.95 ms/token from
`4eaa016b` to `3f412d9b`. Restoring the parent does not recover the alleged
competitive regime. Nevertheless, the candidate's host-sampling architecture
still violates the required device-resident greedy path; treat that as a
correctness-of-design issue, but do not claim it caused a measured competitive
regression from this evidence. Do not start further decode tuning or run a
full curve without a new demonstrated gap.

The P8192 request-window and kernel-family trace attribution above remains the
completed trace result. Its saved launch/query coverage mismatch still blocks
asserting exact P8192 workgroup occupancy/reuse from traces; source-path
comparison is the next bounded attention task, not a reason to rerun traces or
implement another schedule tweak.

### Exact P8192 dispatch reconciliation (2026-09-28)

The earlier geometry caveat is now resolved for mx by inspecting the retained
successful P8192 roctracer capture at
`/tmp/mi50-roctracer-p8192-mx.i6WBzV/2571540_hip_api_trace.txt` (response:
8,192 prompt tokens). It shows `flash_attn_tile<256,256,16,2,false>` with
grid `{x=32,y=3,z=12}`, block `{32,8,1}` (256 threads), followed by
`flash_attn_combine_results<256>` with grid `{x=512,y=24,z=1}`. There are 272
calls to each kernel family in this request window. This is the exact P8192
geometry; it supersedes the earlier `<2,2>`/four-split statement for mx.

| Dimension | MIInfer suffix split-K | pinned mx P8192 | pinned upstream P8192 recapture |
|---|---|---|---|
| Q positions per CTA | 1 | 16 | 16 for 256 main launches; tail kernels use 4 and 2 |
| Q heads per CTA | 2 | 2 | 2 |
| KV head mapping | adjacent pair shares a KV head, but each query token has separate CTA work | two adjacent Q heads share a CTA; each tile is reused across 16 Q positions | same tiled GQA mapping; 16 Q positions and 2 adjacent Q heads in the main tile |
| K/V loads | each CTA streams its token/head's KV prefix; paired heads issue separate per-lane K/V loads | one staged K tile and one staged V tile per CTA/split/loop tile, reused across up to 32 query-head rows | source has same KQ-tile reuse mechanism; exact main dispatch matches `<16,2>` |
| Q/K and V tiling | one query row, 256-d head, online scalar-token loop, unroll 4 | Q tile 16×2; KQ iteration `nbatch_fa=64`, `nbatch_K=128`; V accumulation consumes tile softmax state | trace confirms `<16,2>` main tile; source `DKQ=DV=256`; runtime KQ loop config matches source-selected upstream path |
| Split count | 3 | 3 | 3 for main `<16,2>` launches |
| Reduction | stage 1 partial max/sum/accumulator, separate stage 2 | separate combine kernel over split partials | separate combine kernel over split partials |
| Attention launches in request window | 240 stage-1 + 240 stage-2 | 272 tile + 272 combine | 288 tile + 288 combine (256 main, 16 `<4,2>`, 16 `<2,2>`); 288 combine |
| CTA geometry per stage-1 launch | grid 12×512×3, 64 threads = 18,432 CTAs | grid 32×3×12, 256 threads = 1,152 CTAs | main grid 32×3×12, 256 threads = 1,152 CTAs; tail grids x=1,y=10/4,z=12 |
| LDS / VGPR / spills | selected kernel metadata not retained; do not infer | selected code object: 27,136 B LDS, 97 VGPR, 46 SGPR, zero spills (EXP-0362 build record) | selected code-object values not established for this capture |
| Occupancy | achieved value unavailable | source requests occupancy 3 for `ncols=32`; 27,136 B LDS and 97 VGPR constrain a 256-thread CTA to at most 2 resident CTAs/CU by resource arithmetic; achieved runtime occupancy was not measured | selected object metadata / achieved occupancy not retained; do not infer from launch geometry |
| Intermediate traffic per tile+combine | split workspace stores and rereads 3×(256 FP32 + float2) per query-head result; exact request total needs verified per-call token coverage | for the observed 512 Q positions, 24 heads and 3 splits: 38,043,648 B partial writes + same partial reads + 12,582,912 B final writes ≈84.6 MiB per tile/combine pair | no defensible per-request byte total because geometry does not cover the reported prompt |

The source paths are `ggml/src/ggml-cuda/fattn-tile.cuh` in pinned mx
(`2e9d29fe736969160f17476ecf0a6298cee6966`) and pinned upstream
(`73a43d1f69345aee8bb186ef4b3172cef892f2e5`). In both, `ncols1` is query
positions and `ncols2` is query-head slots. The mx-selected `<16,2>` path
loads a K tile once for `flash_attn_tile_iter_KQ`, reuses it for the 32 Q
columns, then stages/consumes V with the resulting softmax state; for the
Qwen GQA layout those adjacent two Q heads share one KV head. The mx selected
gfx906 config is 256 threads, 27,136 B LDS, 97 VGPR, 46 SGPR, zero spills.
The requested launch occupancy is three, but LDS/VGPR resource arithmetic
caps residency at two CTAs/CU; this is an upper bound, not a measured achieved
occupancy. The trace's combine grid covers 512×24 outputs and its 39,636,224
B allocation is consistent with a max-split partial workspace.

The earlier upstream trace at
`/tmp/mi50-p8192-trace-v2-upstream.jlbkwf/2759307_hip_api_trace.txt` is
retained but invalid for P8192 geometry: every launch was `<2,2>` with
grid `{x=1,y=4,z=12}`, inconsistent with its reported 8,192 evaluated tokens.
Because that capture was demonstrably unusable, one exact-prompt recapture was
made with the pinned upstream build. The new response evaluated 8,193 tokens
(the rendered 8,192-token prompt plus the BOS token), and its API trace is
retained at `/tmp/mi50-upstream-p8192-retrace.gNsv41/server-trace.log` (PID
2938715). It confirms the same main `<16,2>` tiled path as mx: 256 launches
with grid `{x=32,y=3,z=12}`, block `{32,8,1}`, plus 16 each of `<4,2>` and
`<2,2>` tail launches; there are 288 combine launches. Thus the earlier
upstream mismatch was a capture/dispatch-path artifact, not the P8192
architecture. Exact code-object resource metadata and achieved occupancy were
not recorded for this upstream binary, so those remain unknown.

The original request-window MIInfer capture
(`/tmp/mi50-p8192-trace-v2-miinfer.2PPvwn/`) contains 240 stage-1 and 240
stage-2 launches; stage 1 maps one query position and a two-head pair per
64-thread CTA. A separate later scoped capture has 206 stage-1 plus 206
stage-2 calls in its full process trace, not 412 request-window pairs, and
must not be substituted into the original request attribution. MIInfer thus
launches per query rather than sharing one loaded KV tile across 16 query
positions. The combination of measured
attention-family time (MIInfer 5.099 s vs mx 1.785 s) and mx's demonstrated
query/KV reuse is a credible structural explanation for much of the attention
advantage, though attribution is not a causal A/B. Both pinned references now
confirm the same query/KV reuse principle at P8192; MIInfer's per-query CTA
mapping remains the measured structural difference. This makes a KQ-fragment
dataflow prototype evidence-backed, not a speculative schedule variant.

The reuse principle itself is not unprototyped: EXP-0362 exercised a
16-position × 2-head, 32-KV-row LDS candidate at P8192, but that implementation
spilled 137 VGPRs and was much slower; EXP-0363 removed spills (40 VGPR, zero
spills, 28 KiB LDS for candidate A) yet remained slower because it restaged K
and repeated V work per row group instead of matching mx's KQ-fragment
dataflow. Those results reject those implementations, not query/KV reuse as
an architectural direction. A useful next structural candidate must preserve
one K tile across the query rows, consume tile-local KQ/softmax state before
reusing the workspace for V, and pass compile spill/resource gates before a
single P8192 A/B. No kernel code was changed in this checkpoint.

### V2-0043 follow-up correction (2026-09-28)

The pinned AMD config row for `ncols=32` is
`GGML_CUDA_FATTN_TILE_CONFIG_CASE(256,256,16,256,2,32,128)`: the selected
configuration has `nbatch_fa=32`, `nbatch_K=128`, and derived `nbatch_V=16`.
This supersedes any earlier `nbatch_fa=64` reading; captured P8192 launch
geometry and request-window attribution are unchanged. The V2-0043 isolated
prototype subsequently cleared its ≥15% isolated P8192 gate. A later route
audit found that the initial full-model direct-CLI parity/timing pair did not
select the candidate, which had only been wired into `PrefillV2AttentionLayer`;
those end-to-end values are not candidate evidence. The active
`FullAttentionLayer` route is now wired and tested at P8216: the faster FP16-Q
variant diverged from exact greedy token parity at generated token 36, while
FP32-Q restored isolated accuracy but ran at 0.541x control throughput and was
rejected. Neither variant is promoted. See EXP-V2-0043 for the correction and
measurements.
