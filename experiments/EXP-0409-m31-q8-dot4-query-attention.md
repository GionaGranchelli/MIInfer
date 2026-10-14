# EXP-0409 — Q8 query × Q8 key dot4 attention

**Status:** FOCUSED GPU CORRECTNESS PASSES; KERNEL TIMINGS INCONCLUSIVE; ROCM 7.14 GRAPH-DISPATCH FIX BUILDS; DYNAMIC GRAPH REPLAY SMOKE PASSES; FULL-MODEL VALIDATION PENDING
**Baseline:** `d834d9a` on `perf/m31-perf-recovery-0001`

## Question and hypothesis

Can quantizing each 256-wide query head to signed Q8 once, then using gfx906's
`v_dot4_i32_i8` against the already-Q8 key cache, reduce the QK cost in
long-context attention while retaining acceptable output accuracy?

The current Q8 path converts each key byte to float and performs scalar
floating-point multiply-accumulate work. The candidate changes the QK
representation and instruction schedule to 64 packed integer dot operations
per 256-wide key row. V accumulation, softmax, causal masking, KV storage,
query grouping, and the 32-split contract remain unchanged.

## Evidence and scope

EXP-0404 measured attention at 58.448% of the final 512-token tile at 65K.
EXP-0406 measured attention at 80.3% of the 1,024-token suffix at a restored
73K prefix. The kernel-level ceiling is material. QK is one of the two dense
head-dimension products inside attention; its exact share has not been
separately timed, so this is a focused candidate, not a complete cold-prefill
solution.

The AMD `sdot4` intrinsic is already used by MIInfer's gfx906 Q4×Q8 kernels.
Unlike the rejected EXP-0408 FFN experiment, this directly targets the active
attention frontier and retains the production Q8 KV representation. It adds
query quantization error, which must pass the stated correctness gates before
any runtime integration.

## Independent variable

Use signed Q8 query values with a per-query symmetric scale and packed dot4
for QK. Keep K/V Q8 data and scales, Q tile, KV tile, split count, online
softmax, PV accumulation, and output gating fixed.

## Gates

- **Resource:** gfx906 build; zero VGPR/SGPR spills; record VGPR, SGPR, LDS,
  scratch, workgroup and wave size before timing.
- **Correctness:** on deterministic 64K-capacity Q8 operands at 64 and 512
  queries, require finite output, `max_abs <= 0.01`, and cosine `>= 0.9999`
  against direct Q8 attention. Do not relax these thresholds.
- **Performance:** five interleaved runs each at 512 queries, including query
  quantization. Compare dot4 to both the direct production Split-K path and
  the existing Q8 fragment-reuse kernel. The fragment-reuse path is the
  primary control because it holds the attention schedule fixed and isolates
  query quantization plus integer QK. Require at least 1.20x median speedup
  over that control and no regression against production Split-K. Stop if
  slower or if clock/thermal preflight fails.
- **Product:** only after the focused gates pass, run a matched 8K full-model
  correctness and timing comparison. Do not change production dispatch before
  it passes.
- **Kill:** any spills, invalid output, correctness failure, or failure to
  reach the performance gate rejects the candidate.

## Amdahl expectation

The candidate can only accelerate QK, not PV, GDN, projections, or runtime
work. Even an infinitely fast QK step cannot remove more than its share of the
measured attention cost; that share is not yet instrumented. The first focused
result will decide whether to continue or discard the candidate.

## Execution log

- 2026-10-13: HIP kernel and benchmark pass local ROCm 6.2 syntax-only
  compilation; benchmark C++ also passes syntax-only compilation. This is
  source validation only, not the required ROCm 7.14 build or gfx906 resource
  report.
- 2026-10-13 Machinist preflight mapped ROCm device 0 to `0000:04:00.0` and
  device 2 to `0000:88:00.0`. The production server has `ROCR_VISIBLE_DEVICES=0`
  and the candidate server has `ROCR_VISIBLE_DEVICES=2`. ROCm-SMI showed 85%
  and 97% VRAM use respectively (25 C and 27 C edge temperature), with both
  MIInfer processes alive and the thermal guard process present. No benchmark
  was started because neither MI50 was free for an isolated run.
- 2026-10-13 build passed with ROCm image manifest
  `sha256:622c025824d465faa0f5e59eea4f723845032011df273c3b11e25031965fc428`,
  ROCm 7.14 / AMD clang 23, `gfx906`, Release, and the cached
  `nlohmann_json` source from the existing build. The target
  `miinfer-m31-q8-fp16-tiled` linked. No GPU device was exposed to this local
  compile container.
- Compiler resource gate passed with zero spills. The query quantizer uses
  16 SGPR, 11 VGPR, zero scratch, and zero static LDS. The exact dot4
  attention specialization uses 56 SGPR, 119 VGPR, zero scratch, zero SGPR
  and VGPR spills, and 2 waves/SIMD. Its launch reserves 21,248 bytes of
  dynamic shared memory; the compiler's `LDS Size [bytes/block]: 0` is the
  static LDS field, not the launch-time dynamic allocation. The matching Q8
  fragment-reuse control uses 56 SGPR and 85 VGPR, also with zero spills and
  2 waves/SIMD.
- Generated gfx906 assembly for the exact `<true,true,true,true>` attention
  specialization contains `v_dot4_i32_i8` instructions. This confirms the
  intended instruction is present; it does not establish runtime correctness
  or speed.
- Build recovery findings: the ROCm image does not include CMake, and its
  GitHub FetchContent download stalled at zero bytes. Installing CMake and
  using the already-cached dependency source allowed the local build to pass.
- 2026-10-13 18:22 UTC+2 preflight repeated the same busy state: both MIInfer
  servers remained alive, with 85% and 97% VRAM use on `0000:04:00.0` and
  `0000:88:00.0`. Machinist's cached image digest matches the locally built
  image. No GPU execution was started.
- The linked executable has no unresolved ROCm shared libraries under the
  pinned image and was staged on Machinist at
  `/tmp/m31-q8-dot4-remote/miinfer-m31-q8-fp16-tiled`; local and remote SHA-256
  both equal `e7fff9cd29b940efa0cfc67dbec1a560ad309118ec91f3047e000ec34459c6d0`.
- The benchmark interleaves production Split-K Q8, existing Q8 fragment-reuse,
  and Q8 dot4 paths. The fragment-reuse path is the primary control for the
  1.20x gate because it holds the tile schedule fixed.
- 2026-10-13 18:25 UTC+2 live resource check: both loopback MIInfer servers
  report `miinfer_active_requests 0`, with no established TCP clients. The
  existing processes and campaign guard remain running; their session data
  were not touched. GPU0 (`0000:04:00.0`, the candidate server) uses
  29,326,491,648 / 34,342,961,152 bytes, leaving 5,016,469,504 bytes free.
  GPU2 (`0000:88:00.0`) uses 33,312,763,904 bytes, leaving 1,030,199,248
  bytes free. The 512-query synthetic benchmark allocates approximately 1 GB
  across its Q8/FP16 KV arrays, outputs and split workspace, so GPU0 appears
  to have sufficient headroom while GPU2 does not. No execution was launched
  during this preflight. Next safe check: rerun the active-request/client
  check, then execute the 64-query correctness case on GPU0 before the full
  512-query gate; abort on new activity, insufficient memory, or thermal/clock
  guard failure.
- 2026-10-13 18:27 UTC+2 focused GPU run on BDF `0000:04:00.0` in the pinned
  ROCm 7.14 container. Preflight passed: host load average 0.89/0.82/0.83 on
  56 logical CPUs; GPU0 had 5,016,469,504 bytes free, 27 C junction, and
  925 MHz reported idle SCLK. Active requests were zero and there were no
  established clients before execution. The 64-query correctness case passed:
  dot4 max absolute difference `0.0000185`, cosine `0.9999978`. The 512-query
  case passed: dot4 max absolute difference `0.0000238`, cosine `0.9999980`.
  No non-finite outputs or command failures occurred.
- At 512 queries, five interleaved samples (including query quantization)
  measured production Q8 Split-K `428.128–463.783 ms` (median `462.859 ms`),
  matched Q8 fragment reuse `635.668–648.565 ms` (median `635.819 ms`), and
  dot4 `192.898–198.087 ms` (median `194.712 ms`). Dot4 is `3.265x` faster
  than the matched tiled control and `2.377x` faster than production Split-K,
  exceeding the focused numeric thresholds. At 64 queries, correctness also passed;
  dot4 median was `25.184 ms` versus `87.431 ms` tiled and `53.427 ms`
  production. GPU0 junction reached 37 C after the run; VRAM use returned to
  its pre-run level. The benchmark rechecked zero active requests between the
  two cases and after the run.
- Treat the timing as promising, not yet fully isolated: service request
  counters rose from 171/17 to 175/21 on ports 18080/18081 between preflight
  snapshots, although active-request gauges and established-client checks were
  clear at the sampled boundaries. There was no continuous per-sample clock
  or server-request trace, and no profiler identified the remaining limiter.
  Therefore the measured speedup is inconclusive as a qualified performance
  claim despite passing the numeric threshold. Inspect server logs and repeat
  the five-sample A/B in a quiet window before deciding whether to promote.
- 2026-10-13 18:32 UTC+2 server logs identify the request-counter movement:
  both servers received request IDs 172–175 at 18:27:09.628, 18:27:14.458,
  18:27:25.830, and 18:27:48.923. The last event is at the prior benchmark's
  completion boundary, so exact overlap cannot be determined from the current
  log level. A guarded 512-query repeat began at 18:34:30 with both active
  gauges zero and no established clients, then observed both request counters
  increase by two within 248 ms; it terminated before collecting benchmark
  output. This confirms the endpoints are receiving live campaign traffic and
  the original speed samples cannot be treated as isolated.
- 2026-10-13 18:36 UTC+2 quiet-window check sampled both endpoints for 18 seconds.
  Request counters advanced from 183/29 to 244/90, so no 8-second quiet window
  existed and the benchmark was not started. No server was stopped and no
  session files were touched.
- 2026-10-13 18:43 UTC+2 repeat of the guarded quiet-window check began at
  counters 247/93 and observed 307/153 after 18 seconds, with no 8-second
  quiet period. The latest ROCm 7.14 binary hash was staged and verified inside
  the existing container, but the GPU benchmark was not started. No server or
  saved session was changed.
- Extended the opt-in kernel to read `DevicePrefillState` during HIP Graph
  replay, and passed that state through the matching PrefillV2 dispatch. The
  ordinary kernel specializations retain their compile-time no-state path.
  ROCm 7.14 rebuilt and linked the benchmark target. The graph-state dot4
  specialization remains 56 SGPR / 119 VGPR, zero scratch and spills, and two
  waves/SIMD; the matched Q8 control remains 56 SGPR / 85 VGPR with zero spills.
  Generated gfx906 assembly still contains `v_dot4_i32_i8`. Runtime graph replay
  correctness and the full-model path have not been tested.
- Added `MIINFER_M31_Q8_DOT4_ATTENTION=1` as an opt-in PrefillV2 route for Q8
  attention and dedicated reusable Q8-query/scale buffers in the existing
  workspace arena (about 3.05 MiB at 512 tokens). The default dispatch remains
  unchanged. ROCm 7.14 rebuilt and linked the benchmark target with the
  PrefillV2 workspace and dispatch changes. An 8K full-model correctness and
  timing comparison has not yet run; no production qualification is claimed.
- 2026-10-13 source review confirmed that `/metrics` is counted by
  `http_requests`, and `miinfer_requests_total` is an alias of that HTTP-wide
  counter. Both include monitoring requests. The quiet-window check now uses
  `miinfer_inference_requests_total`, `miinfer_active_requests`, and
  `miinfer_queue_depth`; it ignores the HTTP-wide counters. Across 16 seconds,
  inference totals stayed at 55/2 on ports 18080/18081, and both active and
  queued gauges stayed at zero. ROCm-SMI reported 0% GPU utilization at the
  sample. This shows the previous counter movement was not evidence of
  inference overlap; it does not identify every source of those HTTP requests.
  VRAM remained 85%/97% occupied, so this does not make either GPU available
  for a full-model process.
- Source review found that suffix-graph capture passes `prefill_state` while
  calling `forward` with host `base_position == 0`. The Q8 dot4 dispatch had
  required `base_position > 0`, selecting Split-K during graph capture. The
  opt-in dot4 route now also accepts a non-null dynamic prefill state; ordinary
  cold prefill at position zero still uses its existing route. The suffix graph
  cache records the dot4 selection and recaptures if that setting changes.
  Kernel state reads already supply the replay-time base position, token count,
  and capacity, including position zero's causal bounds.
- The graph-dispatch/cache fix, the existing dot4 benchmark target, and the
  main `miinfer` executable compile and link with the pinned ROCm 7.14 / AMD
  clang 23 toolchain for `gfx906`. The build itself did not use a GPU.
- 2026-10-13 18:51 UTC+2: added a focused GPU graph smoke that captures the
  stateful Q8 dot4 wrapper at host base position zero, then replays it at base
  positions 0 and 64. On Machinist GPU2 (`0000:88:00.0`), both graph outputs
  matched direct dot4 execution bit-for-bit (`max_abs=0`, cosine `1.0`). A
  ROCProfiler kernel trace contains the Q8 query quantizer, the
  `qwen35_kq_fragment_reuse_attention_batch_kernel<true, true, true, true,
  true>` specialization, and the Split-K combine kernel. The server inference
  counters remained at 55/2 with active and queued requests at zero. Junction
  temperature moved from 29 C to 30 C under an 80 C warning / 85 C stop guard.
  No service process or session file was changed. This verifies dynamic state
  in a captured dot4 kernel graph; the full PrefillV2 model graph and matched
  8K comparison remain pending.

### 2026-10-14 guarded 64K full-model check

The opt-in Q8 dot4 path completed the same deterministic 65,536-token prompt
with capacity 69,632 and finite full logits on Machinist GPU2 (`0000:88:00.0`,
UUID `GPU-2152414173497dd3`). It took `499.420 s`; the first and second token
IDs were `16467` and `39427`. Source revision
`13418958b5f5146276dc819d59b35418bb911159`, executable SHA-256
`57ec6d1cd975e2d7d0ba770eb592f8997e49dfe9acfab2ae02caf68ea4f4031f`, and
model SHA-256 `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
match the preceding production-control artifact. The prompt sequence and
capacity also match.

Comparing all 248,320 logits with that production control gave max absolute
error `0.9121723`, cosine `0.9935138`, and a different top-1 token (`16467`
versus `174830`). This fails the existing full-logit correctness gate. Reject
the single-plane Q8 query-dot4 candidate regardless of its timing.

The raw `499.420 s` candidate time versus the earlier production control's
`486.268 s` is not a valid speed comparison because the runs used different
MI50 cards. A same-prompt paired production control on GPU0 is still running;
its result will be recorded separately. A focused one-residual-plane query
candidate is also running on GPU2 against the same production reference. Both
runs use independent GPUs and thermal guards; the dot4 candidate remains
opt-in.

### 2026-10-14 one-plane query-residual check

To reduce query quantization error without changing the attention data path, a
second guarded 64K run enabled `MIINFER_M31_Q8_DOT4_QUERY_RESIDUAL=1` on GPU2.
It processed 65,536 tokens at capacity 69,632 and emitted 248,320 finite logits
in `524.220 s`. Source, binary, model, and input match the dot4/control run.
Peak sampled junction temperature was 70 C; neither warning nor stop fired.

Against the exact stored production Q8 control logits, max absolute error was
`0.934224844`, cosine `0.992028804163`, and top-1 matched (`174830`); the first
two generated tokens also matched. It still fails the full-logit gate and was
4.96% slower than the earlier single-plane dot4 run on GPU2. Reject this
residual variant; do not promote it.

The separate paired production control on GPU0 exceeded 20:51 without producing
output despite 100% GPU busy, so it was stopped. Its container required
SIGKILL after the 10-second SIGTERM timeout. The original matching production
control artifact remains the comparator; no paired timing claim is available.
Guard and stop artifacts for the residual run are under
`/home/machinist/m31-q32-exec-20261014T092831Z/final-q8-dot4-query-residual-64k-20261014T101817Z`.

The paired GPU0 production control did not produce logits or timing. After
20:51 elapsed with the MI50 busy at 100%, it was stopped as an anomalously
slow run. `podman stop --time 10` did not terminate the tagged campaign
container; Podman escalated to SIGKILL (exit 137). The Q8 dot4 candidate had
already completed successfully, and its correctness rejection above uses the
matching production-control logits from the earlier run. Stop/cleanup evidence
is retained alongside the run artifacts in
`/home/machinist/m31-q32-exec-20261014T092831Z/final-q8-dot4-64k-20261014T100302Z`.
Do not treat the failed paired wrapper exit as candidate failure; only the
production control was terminated. The residual-plane candidate remains under
its separate guard and was not touched.
