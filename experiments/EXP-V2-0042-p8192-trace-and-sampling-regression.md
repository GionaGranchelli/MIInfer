# EXP-V2-0042 — P8192 trace attribution and sampler-change decode A/B

**Status:** `LEARN — TRACE ATTRIBUTION COMPLETE; SAMPLER CHANGE NOT THE ~20 MS/TOKEN GAP`

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
| mx | 35,084 | 38,552.376 s | 5,284 | 39.038 s |
| upstream | 35,228 | 44,481.246 s | 5,172 | 44.956 s |

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

The HIP API trace records the actual reference symbol
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

`LEARN`: existing P8192 traces are sufficient; do not rerun them. The actual
reference dispatch provides a concrete 2-query × 2-head reuse schedule to
compare against MIInfer's 1-query × 2-head, three-split schedule. Next inspect
the exact pinned mx and upstream source/dispatch paths and complete the
schedule/resource accounting (including occupancy, intermediate traffic, and
launches). Only if that analysis confirms the reuse mechanism explains the
attention advantage should a narrow P8192 prototype be attempted. No full
curve is authorized for an unproven candidate.

The sampling A/B does not authorize further decode tuning or competitor decode
profiling. First fix the required device-resident greedy fast path; design
non-greedy candidate sampling without full-logits D2H and per-token host state
rebuild. Keep that work separate from the P8192 attention hypothesis.
