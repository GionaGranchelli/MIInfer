# EXP-0358 — M26-C current decode-route differential

**Status:** CLOSED — `ROUTES_NOT_COMPARABLE`; no A–D timing qualification
**Date opened:** 2026-09-20
**Starting commit:** `9180e6f5b41c6ece62d68a2f4c153a4a480429af`
**Route scaffold commit:** `1b5c0d497c378f4b2bd00d14bbc2fe2a5225d3a4`
**Measurement implementation base HEAD:** `7bca95720521be6d8cd17b3961d4b5580eedfa96`
**Scope:** Current code only. M27 remains untouched.
**Current outcome:** M26-CQ closed the route comparison as not defensibly comparable under an accepted numerical contract. Keep the qualified no-preset route canonical; do not report an A–D performance comparison.

## Question

How much of the MI50 P512 decode cost differs between the current legacy
batch-major and interactive layer-major routes after both begin from the same
semantic state? Attribute the difference by whole-token operator family,
dispatch and memory traffic, route selection, and position transition.

This does not compare current code with historical M8/M9 results. No kernel or
selector change is allowed until attribution is closed, except a correctness
fix required to establish equivalent state.

## Stop gate

Stop when at least 90% of the measured route delta is attributed, no removable
family accounts for at least 3 ms/token, the remaining cost is shown to be
required by the current contracts, or equivalent comparison proves impossible
without changing semantics. Do not proceed into M27 work.

## Contract mapping

`Qwen35RuntimeEngine.generate()` selects `generate_fresh()` when
`MIINFER_PREFILL_LAYER_MAJOR` is off and `generate_layer_major()` when it is
on. The former processes `P-1` prompt inputs then consumes the final prompt
token through `step()`; the latter prefills all P prompt tokens and computes
the first output token from final hidden state. For P512, both can reach the
semantic boundary after all 512 inputs have been processed, at position 512,
with the next generated token available. Their built-in timings do not match:
legacy includes the final prompt token in a normal decode step, while
interactive counts the LM-head first-token operation with its decode time.

Legacy direct decode synchronizes and copies one token D2H per step. Legacy
queued graph replays at positions beginning 511 and counts graph launches per
token. Interactive direct starts its post-prefill decode at position 512;
interactive graph replays from there and bulk-copies generated tokens. Graph
capture time is tracked separately. These existing timing fields cannot be
compared as-is; the harness must time a common post-P512 boundary and report
prefill/first-token/graph setup separately.

`--check-graph-state` already runs graph then direct in one runtime and checks
token parity plus byte-identical recurrent state, convolution history, and
active K/V snapshots. `snapshot_decode_buffers()` has no corresponding import
helper, and session checkpoints reject differing runtime identities. Thus no
existing cross-route state restore is available.

The `m25_interactive` preset changes more than scheduling: it selects wide,
full-layer-major Mx prefill/repacked resident weights, Mx GDN prefill, wide Mx
attention decode, and `MIINFER_MX_MMV=1`. The latter selects one-token Mx MMV
dispatches in `gfx906/kernels/kquant_wave_layout.hip`. Constructor-owned layer
configuration and weight representation prevent toggling the route safely on
one already-created engine. A serial reload/import diagnostic is required to
avoid two simultaneous model copies on the MI50.

## Initial contract matrix

| ID | Route | Decode execution | Status |
|---|---|---|---|
| A | legacy / batch-major | direct step | Supported by current selector |
| B | legacy / batch-major | queued HIP graph | Supported by current selector |
| C | interactive / layer-major | direct step | Technically selectable; verify exact supported execution |
| D | interactive / layer-major | queued HIP graph | Current no-stream interactive path |

Run only contracts that can start from the same post-P512 semantic state.
Record unsupported combinations and why. Do not silently substitute.

## Required measurements

1. Establish one canonical P512 state and next token. Compare imported state
   snapshots and next-token/logit behavior before timing. Include position,
   token, KV dtype/layout, recurrent layout, convolution history, and all
   active KV entries in the equivalence record. Keep restore/setup outside the
   steady-state timing boundary.
2. Run A–D where supported, with matched P512 state, generated-token count,
   model/build, selectors, warmup, and fixed MI50 clocks at SCLK/MCLK
   `1606/1000 MHz`. Report first-token, steady decode, graph capture, and
   preparation separately; retain raw outputs.
3. Attribute one complete token by semantic family (recurrent projections,
   GDN/state, FFN, attention/QKV/KV/attention O, norms/conversions, LM
   head/argmax, dispatch/synchronization/runtime). Avoid summing overlapping
   spans. Record dispatches, synchronizations, H2D/D2H bytes, allocation and
   VRAM deltas, and available GPU memory counters per route.
4. Audit route selection from process entry through constructor flags and
   kernel launch. Record active/replaced/disabled/unreachable selector status.
5. Verify token and state equivalence across the boundaries 512→513,
   2048→2049, and 8192→8193. Record any capacity or graph-support limits.
6. Sample hardware continuously with `scripts/sample-gpu.sh`. Any material
   SCLK/MCLK deviation from 1606/1000 makes the run diagnostic only; preserve
   it with the telemetry rather than silently discarding it.

Use interleaved A/B order and repeated runs for timing conclusions. Preserve
raw run logs, telemetry, exact command/environment, model hash, compiler,
ROCm, VRAM, temperature, and clocks.

## Diagnostic result (not timing qualification)

The diagnostic harness uses one canonical P512 snapshot made by the legacy
route: position 512, pending token 20, exact model SHA256
`7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`. It
serializes processed token history, generated IDs, each recurrent logical FP32
state and convolution history, and every active attention K/V entry in
canonical FP32 form. Restore converts state to the target route's native
representation. A restore-only legacy→interactive round trip compares exactly
across all serialized semantic fields. Restore is outside the decode timer.

All four routes A–D imported that same snapshot and decoded 128 tokens. The
four output token sequences were identical (the repeated synthetic P512 prompt
produces a 14-token cycle); the first generated token was `15` in all routes.
Final position was 640 with pending token `7554`. Within each execution route, direct and graph
outputs and final semantic state were byte-identical. Cross-route final state
was not byte-identical despite equal generated token IDs: legacy vs interactive
maximum absolute differences were 0.429274 in recurrent state, 8.335708 in
convolution history, 21.494141 in active K, and 60.064453 in active V. The
existing contract establishes direct/graph state parity within one route; it
does not define a cross-route internal-state tolerance. Therefore the same
token sequence is insufficient to close cross-route correctness. This
correctness boundary remained open at the time of this probe; no tolerance
was added. M26-CQ later closed the route comparison as `ROUTES_NOT_COMPARABLE`
under the available numerical contracts; see the final re-evaluation below.

At TG1, the divergence already spans recurrent layers, attention K/V, and
convolution histories (max abs 0.190060, 3.157227, 12.968750, and 5.534039
respectively). Layer-level errors begin small and grow through later blocks;
this is not explained by snapshot restore, whose legacy→interactive
restore-only round trip is exact. The M25 scalar-versus-wide diagnostics
reported 0.00230 recurrent-state and 0.00702 convolution-history maxima, but
those are observations, not an accepted cross-route tolerance. M9 defines an
external-logit criterion: 64/64 teacher-forced argmax agreement and logits
cosine at least `0.9995` against its CPU/high-precision oracle at position 64.
That oracle and position-specific result do not define a tolerance for
legacy-versus-interactive state at P512 or for free-running TG128. No current
accepted criterion permits this state difference to be called equivalent.

The repeated-token seed is a weak token-parity probe: its TG128 output forms a
14-token cycle. A second canonical P512 snapshot was prepared from the first
512 tokens of committed varied text in EXP-0351 (prompt SHA256
`e634bbe0a258cb5a51b7b6e557aea5d04e659d0b5d187ec841ca662fc6c86f19`, pending
token `2193`). Legacy and interactive direct decode both emitted first token
`1406`, but diverged at generated-token index 55 (legacy `16`, interactive
`15`). The processed input histories first differ at absolute index 568. After
128 tokens, state max_abs differences were 2.871509 recurrent, 20.115730
convolution history, 20.744141 active K, and 103.960938 active V. This varied
prompt therefore fails cross-route token equivalence; its single-run timings
(32.749 vs 59.433 ms/token) are not a matched TG128 comparison and are not
used for route-delta claims. The first-step state difference was much smaller
than the repeated-token fixture, confirming that error growth depends on the
prompt/state. The full 128-token state comparison is retained as
`legacy-vs-interactive-varied-tg128.compare.log` beside the snapshots. A fresh
same-build one-token instrumented capture localizes the
first recorded varied-prompt divergence to recurrent layer 0: max_abs is
`0.000592232` in recurrent state and `0.00115204` in convolution history;
active K/V diverges later at attention layer 3. The canonical state at position
512 and pending token `2193` is shared, and both routes emit `1406`; the
difference is introduced during the first route-specific decode, not by
snapshot restore. Source tracing narrows the first convolution-history delta
to the recurrent QKV projection: `conv_silu_split` consumes `qkv_dest`, while
the layer gate, beta, and decay projections do not feed convolution history.
Legacy uses the combined native QKV+gate Q4 wave GEMV path; resident-all
interactive decode quantizes to Mx Q8 and uses the resident repacked Q4 MMQ
path. The `m26_mx_mmq` control retains that repacked QKV path and reproduces
nearly the same convolution-history error; disabling resident-all restores
bitwise layer-0 convolution history. This attributes the first history delta
to the selected QKV backend/representation. It does not prove how much of the
separate recurrent-state delta comes from QKV versus gate or recurrence.

### Existing selector-control probe (diagnostic only)

The existing `m26_mx_mmq` selector-control preset (same interactive/layer-major
setup as `m25_interactive`, with `MIINFER_MX_MMV` unset) was restored from the
same varied P512 snapshot and decoded one token. It also emitted `1406`. Its
comparison against legacy already has recurrent layer-0 maxima `0.000593185`
(state) and `0.00116730` (convolution history), close to interactive MMV's
`0.000592232` and `0.00115204`. Comparing the two interactive selector arms,
the layer-0 differences are only `1.91e-6` and `1.53e-5`; their recurrent-state
and convolution-history differences grow later, with the largest recurrent
difference first appearing at layer 2. This rules out the MX_MMV selector as
the cause of the first layer-0 mismatch against legacy. It does not identify
which other wide/layer-major route choice causes that mismatch.

This selector-control run is a numerical diagnostic, not a valid A-D timing
cell: it is outside the production `m25_interactive` execution contract and
the existing M26-B selector study already found substantially different full
generation behavior. A further attempt to disable resident-all weights while
leaving Mx attention decode enabled was rejected by the runtime with
`Mx attention decode requires resident-all attention weights`; the combination
is unsupported and was not measured. The TG1 state and full comparator outputs
are retained as `mx-mmq-direct-varied-tg1.state`,
`legacy-vs-mx-mmq-tg1.compare.log`, and
`mx-mmq-vs-mx-mmv-tg1.compare.log` in the external artifact directory.

A second one-token probe disabled `MIINFER_PREFILL_REPACKED_RESIDENT_ALL` in
the same wide/layer-major configuration and also disabled
`MIINFER_PREFILL_WIDE_MX_REPACKED_ATTN_DECODE`, which the runtime requires
when resident-all is off. This selected the legacy Q8/Q4 QKV+gate projections
for recurrent layers. The resulting layer-0 recurrent state and convolution
history were bit-identical to legacy; divergence first appeared at recurrent
layer 1 (`0.0402448` recurrent state, `1.02923` convolution history), and the
first generated token changed to `2100` (legacy: `1406`). The run therefore
supports the resident-repacked recurrent projection path as the source of the
earliest state delta, but cannot isolate QKV from gate because both branch
together, and cannot validate the production interactive route because Mx
attention decode was disabled. The altered load allocated 22,279,885,204 bytes
(2,493 allocations), 3,461,087,232 bytes more than legacy. This is a controlled
correctness probe only, not a supported A-D cell or performance candidate.
Its state and comparison are `wide-nonresident-direct-varied-tg1.state` and
`legacy-vs-wide-nonresident-tg1.compare.log`.

### M23 resident-repack selector control (diagnostic only)

To distinguish the Mx resident format from the legacy native Q4 path, one
additional TG1 direct run kept layer-major/full-layer execution, resident-all
weights, and the wide repacked MMQ paths enabled, but set
`MIINFER_PREFILL_WIDE_MX_REPACKED_MMQ=0`. The recurrent QKV dispatch therefore
uses M23 Q8 activation quantization with repacked Q4 MMQ weights. It emitted the
same token `1406`. Against legacy, layer-0 recurrent state and convolution
history maxima fell to `3.81e-6` and `1.91e-5`, from `5.92e-4` and `1.15e-3`
with the Mx resident path. They are still not bitwise equal. The first larger
recurrent/conv divergence is layer 2 (`1.02e-4` / `0.00212`); active attention
K/V divergence starts at layer 3. This supports Mx QKV quantization/format as
the source of most of the immediate layer-0 error, without establishing an
accepted tolerance.

This switch is global across Mx resident projections, not a QKV-only selector,
so later recurrent and attention differences are confounded and it is not an
A-D cell. CPU-reference logits still agree on top-1 `1406` but cosine is
`0.999377290` (max_abs `0.422607`, RMS `0.081531`), below the M9 `0.9995`
position-64 floor when applied diagnostically at this frontier. The single-run
decode duration was `414.65 ms`; it is diagnostic only and not comparable as a
performance sample. Release binary SHA256 is
`0f55bdcee3df810997d89474985aad70fb1a961fc5a918a23da93d124e99e7c4` at HEAD
`8bf590a2d21023c5be86c578d59c833344c131ef`. Raw state, logits, process log,
and CPU-logit comparison are retained as `interactive-m23-repacked-oracle.state`,
`interactive-m23-repacked-varied-oracle.logits.f32`,
`interactive-m23-repacked-oracle.log`,
`legacy-vs-m23-repacked-direct-varied-tg1.compare.log`,
`interactive-mx-vs-m23-repacked-direct-varied-tg1.compare.log`, and
`m23-p512-logit-oracle-comparison.log` in the external artifact directory.

### Recurrent-QKV-only M23 control (rejected)

The global M23 control nearly restored layer-0 recurrent/conv state, so an
opt-in `--m26c-recurrent-qkv-m23` diagnostic selector was added to change only
the resident recurrent QKV weight repack and Q8 format while retaining the
interactive Mx paths for gate, SSM-out, FFN, and attention. It requires an
imported M26-C snapshot and Mx resident-all configuration; the selector is
included in the execution-contract identity. Only one QKV representation is
resident, so it adds no duplicate model-weight copy.

From the same varied P512 snapshot, this mixed-format route produced NaN/Inf
logits. The argmax buffer contained `248320`, equal to the vocabulary size
`248320`, and the M26-C snapshot writer rejected it as an invalid current
token. Direct inspection found all `248320/248320` captured logits were NaN
(zero finite values and zero infinities); the generalized logit comparator
also rejects the vector. This is an invalid mixed route, not an A-D cell or a
state comparison; no timing or correctness claim is drawn from it. The captured
binary SHA256 was
`ea7626d481140aa850cc4e0f5df84e675c830e267a89488172c2779c792e788e`. Raw
process output, invalid logits, and comparator rejection are
`interactive-m23-qkv-only-direct-varied-tg1.log`,
`interactive-m23-qkv-only-varied-oracle.logits.f32`, and
`m23-qkv-only-logit-oracle-rejected.log` in the external artifact directory.

### CPU-oracle logits at the common frontier

The exact 512 token history was fed to the pinned llama.cpp CPU reference at
`c0bc8591e8815c63cb01dd3f051a8b0df02501c9`; a byte comparison confirms it
matches `p512-varied.state`. The reference's first generated token is `2193`,
also the snapshot's pending input. After consuming `2193` at absolute position
512, the CPU reference and both GPU routes emit `1406` as argmax. The legacy
and interactive position-512 logits have cosine `0.999773389` to each other.
Against the CPU logits, legacy cosine is `0.999381363` (max_abs `0.439167`,
RMS `0.080973`) and interactive cosine is `0.999245820` (max_abs `0.484208`,
RMS `0.089351`). Thus both agree on top-1 at this position, while both are
below the M9 `0.9995` cosine floor if that floor is applied here. M9's formal
fixture covers its own P12/position-64 contract; it has not qualified this
P512 frontier, so this is reported as a one-position diagnostic rather than
an M9 pass/fail result. The generalized
`scripts/compare-m26c-logits.py CPU.f32 NAME=ROUTE.f32 ...` comparison reports
all three route logits against the same oracle and pairwise; M23-vs-legacy
cosine is `0.999884266`, compared with legacy-vs-interactive `0.999773389`.
It still does not resolve the TG128 token divergence
at generated index 55 or internal-state differences.

The current Release executable used for all GPU logit captures has SHA256
`0f55bdcee3df810997d89474985aad70fb1a961fc5a918a23da93d124e99e7c4`. Raw
CPU/GPU logits and the comparison output are retained in the artifact directory as
`m26c-p512-cpu-reference/`, `legacy-direct-varied-oracle.logits.f32`,
`interactive-direct-varied-oracle.logits.f32`,
`interactive-m23-repacked-varied-oracle.logits.f32`, and
`p512-logit-oracle-comparison.log`.

Single diagnostic TG128 timings (one run each, not qualified, no optimization
claim):

| Route | Decode ms | ms/token | Graph capture ms |
|---|---:|---:|---:|
| A legacy direct | 4203.523 | 32.840 | — |
| B legacy graph | 4133.964 | 32.297 | 17.205 |
| C interactive direct | 7701.139 | 60.165 | — |
| D interactive graph | 7456.355 | 58.253 | 17.678 |

The matched direct pair gives the diagnostic route delta
`C - A = 60.165147 - 32.840022 = +27.325125 ms/token`. This is one sample
per route and is not a qualified estimate. The graph pair is not compared:
route B has a coincident 1485 MHz SCLK telemetry sample and is excluded.
The interactive direct/graph single-run difference is `-1.912377 ms/token`,
but it combines graph replay and bulk-copy behavior and is not an isolated
graph benefit measurement.

The direct one-token profiler is attribution evidence only. Inspection found
the existing attention stage labels were shifted after Q norm/RoPE; the
diagnostic label table was corrected and both profiles rerun. In the corrected
profiles, wall times were 50.095 ms legacy and 76.093 ms interactive.
Selected layer events accounted for 38.344 and 64.194 ms, respectively
(25.850 ms delta, 99.4% of the 25.997 ms wall delta). Largest family deltas
were attention Q/K projection +8.897 ms, attention V projection +8.355 ms,
recurrent FFN gate/up +4.171 ms, recurrent gate projection +1.451 ms, and
attention FFN gate/up +1.387 ms. The prior label-based attribution is
superseded. Nested ordered-path spans overlap family events and are not summed.

The measured direct selected-token route-delta table uses those instrumented
event runs. Shares are family delta divided by the 25.997 ms outer wall delta;
event timing is diagnostic and not throughput qualification.

| Family / contract | Legacy ms/token | Interactive ms/token | Delta | Route-delta share | Classification | Confidence |
|---|---:|---:|---:|---:|---|---|
| Attention Q/K projection | 1.150 | 10.047 | +8.897 | 34.2% | ROUTE_IMPLEMENTATION | Medium |
| Attention V projection | 0.275 | 8.629 | +8.355 | 32.1% | ROUTE_IMPLEMENTATION | Medium |
| Recurrent FFN gate/up | 7.041 | 11.212 | +4.171 | 16.0% | ROUTE_IMPLEMENTATION | Medium |
| Recurrent gate projection | 1.002 | 2.453 | +1.451 | 5.6% | ROUTE_IMPLEMENTATION | Medium |
| Attention FFN gate/up | 2.335 | 3.723 | +1.387 | 5.3% | ROUTE_IMPLEMENTATION | Medium |
| Recurrent SSM output projection | 2.155 | 2.741 | +0.586 | 2.3% | ROUTE_IMPLEMENTATION | Medium |
| Recurrent FFN down | 5.055 | 5.565 | +0.510 | 2.0% | ROUTE_IMPLEMENTATION | Medium |
| Other measured families, net | 19.332 | 19.825 | +0.493 | 1.9% | UNATTRIBUTED | Low |
| Outer wall minus accounted events | 11.751 | 11.898 | +0.147 | 0.6% | UNATTRIBUTED | Low |

The large families are classified as ROUTE_IMPLEMENTATION because the source
confirms different selected projection backends; this does not establish that
any work is redundant or removable. Interactive attention uses resident
repacked Q/K and V projections, while legacy uses native Q4 Q/K and V paths.
The interactive recurrent route selects Mx repacked MMV for one-token FFN
gate/up projections. The profile names identify semantic stages, not
individual kernel launches. Graph node counts are 976 legacy / 1,232
interactive, a difference of 256 nodes per replay; node types are not yet
separated. Existing profiles show no large extra route-specific restore cost
inside the decode boundary. Granular launch/copy counters are still needed
before assigning the differences to a specific launch, conversion, or
synchronization. By coverage, 97.5% of the measured route delta is classified
as ROUTE_IMPLEMENTATION, while 2.5% remains UNATTRIBUTED; selected stage events
account for 99.4% of the outer wall delta. This clears the experiment's 90%
diagnostic attribution-coverage threshold only. These are single-run profiling
data, and the table does not identify a correctness-preserving, removable
family, so it authorizes no performance candidate.

## Dispatch and memory audit

Source-path inspection confirms:

- Each attention route has one K/V cache and one set of active entries; each
  recurrent layer has one semantic GDN state/history pair. Restore performs
  one copy into that route's native state/cache representation; no duplicate
  semantic state representation is created.
- The graph's extra 64-byte `DeviceDecodeState` is token/position control
  metadata, not a copy of recurrent/KV state. Graph topology is built once
  before timing and replayed without per-token graph checks or workspace
  rebuilds. Device workspaces are allocated at load; no device allocation is
  in the decode loop.
- Legacy direct uploads one 4-byte input token and downloads one 4-byte output
  token per step, then synchronizes that step. Interactive direct has the same
  host transfer/sync contract. Graph routes upload the 64-byte control state
  once, replay 128 times, and bulk-download 512 result bytes with one blocking
  result transfer. These are byte/call counts; individual copy and wait time
  is not separately measured. For graph routes, control-state upload and its
  initial stream synchronization occur before `decode_ms`; graph replay and
  bulk output transfer occur inside it. Direct per-token copies and waits are
  inside `decode_ms`.
- In attention, legacy's combined Q/K projection quantizes normalized input
  once and reuses Q8 for V. The interactive resident-repacked Q/K path and V
  path each quantize the same normalized input for their M23 layouts, adding
  one observed source-level quantization per each of 16 attention layers and
  per token. No equivalence of those quantizer formats is assumed.
- In recurrent blocks, the interactive repacked path splits projections that
  the legacy selector can combine; interactive FFN Gate/Up also selects
  separate Mx MMV projection launches followed by SwiGLU. This matches the
  large attributed stage families, but does not prove the work can be removed
  while preserving the output contract.

The corrected profile logs report 2,397 device allocations / 18,785,765,812
bytes for legacy and 2,637 allocations / 18,472,649,108 bytes for interactive.
Interactive has 240 more load-time allocations but 313,116,704 fewer allocated
bytes in this build. Allocation counts include model-specific weight and
workspace buffers; per-operator bytes are not currently separated.

The canonical restore transfers 192,413,696 semantic bytes H2D, outside decode
timing. For 128 direct tokens the path copies 512 bytes H2D and 512 bytes D2H
over 128 steps, synchronizing once per token. Graph decode copies one 64-byte
`DeviceDecodeState` H2D and 512 output bytes D2H, with 128 graph replays and a
bulk result synchronization; graph capture was ~17 ms and measured separately.
One-token graph captures contained 976 nodes for legacy and 1,232 for
interactive. This is a total graph-node count, not yet a kernel-only dispatch
count. No per-token device allocation is present in the decode loop.

The original telemetry file contains 7,266 samples: MCLK was 1000 MHz in all
samples, while SCLK was 1606 MHz in 7,265 and 1485 MHz in one sample at
16:57:31.474 UTC, with 19.56 GB VRAM in use. That sample overlaps the legacy
graph TG128 run window (its state file was written at 16:57:32 UTC); mark B
clock-contaminated and exclude its timing. A later telemetry window for the
relabelled attribution profiles had 661/661 samples at 1606/1000 MHz. The
TG128 values remain diagnostic single runs, not interleaved, repeated timing
conclusions. Five clean interleaved samples per comparable cell, full run
metadata, and the cross-route state contract are still open.
The earlier EXP-0353 boundary checks at 512→513, 2048→2049, and 8192→8193
were rerun with the current Release build and all passed graph/direct token
and full-state checks:

| Start → end | Result | Compared state bytes |
|---|---|---:|
| 508 → 513 | PASS | 192,479,232 |
| 2,044 → 2,049 | PASS | 293,142,528 |
| 8,188 → 8,193 | PASS | 695,795,712 |

The 8K case used `m25_hi_qualified`; the shorter cases used default selectors.
Raw boundary logs, both P512 snapshots, route state outputs, profiles, graph
summaries, and telemetry are in
[the external EXP-0358 artifact directory](/home/fedora-workstation/Development/mi50-artifacts/exp0358-current-route/20260920/README.md).
No M27 conclusion is changed.

### Follow-up: copy/synchronization wall timers

The current diagnostic build adds direct-route HIP events around the 4-byte
H2D/D2H copies and a host timer around each explicit stream synchronization.
For graph routes it reports the pre-decode state synchronization, replay enqueue
wall time, and blocking bulk-copy wall time separately. The graph output vector
is resized before the decode timer, matching the direct route's pre-timer
reservation. These counters add instrumentation overhead and are not throughput
qualification.

One-token runs from the varied P512 snapshot all emitted token 1406. Single-run
diagnostic breakdown:

| Route | Decode ms | H2D device ms | D2H device ms | Explicit sync ms / calls | State sync ms | Replay enqueue ms | Bulk copy + wait ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Legacy direct | 42.592 | 0.0152 | 0.0160 | 0.0255 / 1 | — | — | — |
| Interactive direct | 66.500 | 0.0446 | 0.0174 | 0.0124 / 1 | — | — | — |
| Legacy graph | 30.842 | — | — | — | 0.0399 | 0.0894 | 30.753 |
| Interactive graph | 56.025 | — | — | — | 0.0364 | 0.1071 | 55.918 |

The graph bulk-copy timer measures synchronous copy plus waiting for prior
graph work, not isolated D2H transfer cost. These four instrumented single-token
samples only locate the graph's decode wait and show that explicit copy/sync
timers do not explain the direct decode premium; they do not isolate direct
launch/dispatch cost or qualify route timings. Output snapshots are retained in
the artifact directory with `-varied-transfer` filenames. The remaining timing
and correctness gates stay open.

The same-route direct/graph state comparisons for each contract both PASS
bitwise at position 513, including generated token ID and all recurrent,
convolution, and active KV state. The direct cross-route comparison remains a
FAIL for exact state, despite both routes emitting token 1406; that numerical
difference is the previously established route divergence, not a new graph
parity failure.

The varied-state one-token A-D cross-check confirms the dispatch split:
legacy direct versus legacy graph and interactive direct versus interactive
graph are bitwise identical at position 513. Legacy versus interactive has the
same aggregate maxima with either dispatch route (recurrent state `0.0146670`,
convolution history `0.287508`, active K `0.158203`, active V `0.329102`), while
both emit token `1406`. Thus the first semantic mismatch follows the selected
route/backend contract and is not introduced by graph replay. Full comparator
outputs are retained as `legacy-direct-vs-graph-varied-tg1.compare.log`,
`interactive-direct-vs-graph-varied-tg1.compare.log`,
`legacy-vs-interactive-direct-varied-tg1.compare.log`, and
`legacy-vs-interactive-graph-varied-tg1.compare.log` in the external artifact
directory.

M26-B closed at commit `9180e6f5b41c6ece62d68a2f4c153a4a480429af` after
proving its historical comparison non-comparable; its result is not a current
route baseline.

The route mapping/documentation scaffold was committed separately at
`1b5c0d497c378f4b2bd00d14bbc2fe2a5225d3a4`. The measurement implementation
started from the subsequent clean documentation commit
`7bca95720521be6d8cd17b3961d4b5580eedfa96`; no harness or performance-candidate
changes were included in either starting commit.

## Decision

Prior open decision (superseded by M26-CQ): the canonical position-512 semantic state restores into both current
routes, and same-route direct/graph parity is exact. Cross-route correctness
fails on the varied prompt: both routes emit token 1406 first, but layer-0
convolution history differs immediately and generated token IDs first diverge
at index 55. The mismatch is consistent across direct and graph routes, so
graph replay is not its cause. Mx resident QKV representation is implicated:
the M23 resident-repack control reduces layer-0 recurrent/conv error from
`5.92e-4` / `1.15e-3` to `3.81e-6` / `1.91e-5`, but its global selector
changes other resident projections and it does not meet the diagnostic CPU
logit cosine floor. The QKV-only M23 control produced nonfinite logits and an
out-of-range argmax, so that mixed route is rejected. The alternate
nonresident probe that restores exact
layer-0 history also disables Mx attention decode, changes the first output
token, and adds 3.46 GB of load allocation; neither probe is a valid route
correction. No accepted cross-route internal-state tolerance exists.

The one-position P512 CPU-logit diagnostic gives matching top-1 token 1406
for legacy, interactive, and M23 repack, but all three cosines fall below
`0.9995` if the M9 floor is applied diagnostically. This is not the formal M9
P12/position-64 gate; its reconstructed fixture uses a different prompt and
cannot qualify this frontier. Profile attribution classifies 97.5% of the
measured route delta as ROUTE_IMPLEMENTATION and leaves 2.5% UNATTRIBUTED, but
the profile data are single-run and do not show a correctness-preserving,
removable family. Attribution coverage therefore clears 90%, while the
correctness and candidate-removability gates remain open.

TG128 A-D timing cannot be qualified; all current timings remain diagnostic.
The exact common semantic frontier is P512 with pending token 2193; both
production routes emit token 1406 after that input, then mutate semantic state
differently and diverge in token IDs at index 55. Continue with a
correctness-preserving route diagnosis before repeated timing qualification;
do not select a performance optimization or close M26-C.

### Re-evaluation — isolated recurrent QKV backend control

The initial QKV-only M23 probe above was not valid: it wrote M23-format Q8
activation blocks for QKV, then the following Mx gate projection consumed the
same buffer as Mx-format blocks. The all-NaN output was caused by that diagnostic
selector's format handoff, not evidence against M23 QKV. The M26-C-only path now
re-quantizes the shared normalized activation into Mx Q8 before the gate. The
normal decode route is unchanged. This supersedes the earlier interpretation
that the all-NaN capture rejected M23 QKV itself; only the original invalid
mixed-format handoff is rejected.

With the corrected handoff, one Release TG1 diagnostic from the same varied
P512 snapshot was finite (248320/248320 logits) and emitted token `1406`, the
same first token as both production routes. Its position-513 logits were not
identical to either production route. Against legacy,
layer-0 recurrent-state/history errors were `3.81469727e-6` and
`1.90734863e-5`; this is substantially smaller than the original Mx-QKV
interactive errors (`5.92232e-4` and `1.15204e-3`) and matches the global M23
repack probe at layer 0. The full state still differs: recurrent `0.0129924`,
convolution history `0.216493`, active K `0.191895`, active V `0.362549` max
absolute error against legacy at position 513. It therefore does not establish
cross-route semantic equivalence. CPU-logit cosine was `0.999327976`, below
the diagnostic `0.9995` reference floor; that floor is not a formal P512 gate.

The run's single `94.310657 ms` decode time is diagnostic only. The 376
continuous telemetry samples all reported 1606/1000 MHz. The output is retained
as `interactive-m23-qkv-only-gatefix-direct-varied-tg1.log`,
`interactive-m23-qkv-only-gatefix-varied-tg1.state`, and
`interactive-m23-qkv-only-gatefix-varied-oracle.logits.f32`; legacy and
interactive state comparisons plus the CPU-logit comparison are the matching
`*-qkv-only-gatefix*` files in the external artifact directory. This control
identifies the first layer-0 convolution-history delta as specific to the
recurrent-QKV representation/backend selection, but the later state and KV
deltas remain unexplained. No A-D timing qualification or optimization is
authorized by this result.

### Layer-stage capture — first hidden-state divergence

To localize the remaining drift, the M26-C direct TG1 diagnostic can attach
the existing `LayerPathCapture` and `GatePathCapture` hooks to recurrent layers
0 and 1. It dumps inputs, normalized inputs, QKV, recurrent output, gate
projection, gated output, residual, post-normalization, FFN output, and final
layer output. The captures add host downloads and are diagnostic only. Against
the same binary's uninstrumented decode, all recurrent state, convolution
history, and active KV bytes remain exactly equal for legacy, interactive, and
QKV-M23. Each route starts at position 512 with token `2193` and emits `1406`.

All three capture runs used Release binary SHA256
`3dbbf35a63b827b98215d9ab85a97d530dad41690ed483f43bb6bef2b51256b1`. The
294 legacy, 347 interactive, and 374 QKV-M23 telemetry samples each report
1606/1000 MHz. These are not performance samples.

| Layer-0 capture | Legacy vs interactive max abs | Legacy vs QKV-M23 max abs |
|---|---:|---:|
| Normalized input | 0 (exact) | 0 (exact) |
| QKV | 0.00115204 | 0.0000190735 |
| Gate projection | 0.0243430 | 0.0243430 |
| Recurrent output | 0.0000491142 | 0.000000476837 |
| Gated output | 0.000654250 | 0.000654325 |
| Residual after SSM output projection | 0.0666714 | 0.0667992 |
| Post-normalized residual | 0.0304382 | 0.0304174 |
| FFN output | 0.00694430 | 0.00463966 |
| Layer output / layer-1 input | 0.0736160 | 0.0714388 |

The QKV-M23 control reduces the QKV and recurrent-output deltas at layer 0,
but the main hidden activation delta remains after SSM output projection. The
input to that residual is bitwise equal at layer 0, so its residual difference
tracks the selected SSM-output route: interactive takes the resident Mx
Q5_K/MMQ branch, while legacy takes native Q5_K wave GEMV. The Mx gate
projection also differs from legacy (`0.0243430` max abs), but is identical
between interactive and QKV-M23; the capture shows its gated-output difference
is smaller (`0.0006543`). Layer-0 output becomes layer-1 input (`0.0736160`
legacy/interactive; `0.0714388` legacy/QKV-M23), explaining why later
projection/state differences cannot be attributed to QKV alone. FFN also
contributes a measured layer-0 output difference. This is source-correlated
single-token numerical attribution, not an accepted tolerance or a removable
performance family. No route equivalence or optimization is established.

Raw tensors, manifests, logs, telemetry, route-pair comparisons, and
instrumentation-state validation are retained as `*-layerpath*` artifacts in
the external artifact directory. `scripts/compare-m26c-layer-path.py`
reproduces the vector comparisons.

### Native SSM-output selector control

The Mx resident-all selector forced the interactive recurrent SSM-output
projection through resident Mx Q5_K/MMQ despite
`MIINFER_Q5K_NATIVE_SSM_OUT=1`; that native selector is gated off when
`resident_repacked_all` is active. An imported-state-only
`--m26c-ssm-out-native` diagnostic control now selects the already-existing
native Q5_K wave-GEMV path for SSM output while leaving QKV, gate, FFN, and all
other Mx resident families unchanged. It omits the resident Mx SSM-output
weight copy when enabled, so the two representations are not retained
together. The normal route is unchanged when the flag is absent.

Legacy, interactive, and native-SSM captures used the same Release binary
SHA256 `10b2ddb840123159defec1f004be72a4f29017f7c7260d50b6f5e9fcbd0f0f1e`,
same P512 state (position 512, input token `2193`), and all emitted `1406`.
Telemetry was 309, 451, and 423 samples respectively; every sample reported
SCLK/MCLK `1606/1000 MHz`. Interactive and native-SSM layer-0 captures are
bitwise equal through QKV, recurrent output, gate projection, and gated output;
they diverge at the SSM-output residual as expected from the single changed
family.

| Comparison | Layer-0 residual after SSM output | Layer-0 output / layer-1 input |
|---|---:|---:|
| Legacy vs interactive Mx SSM output | 0.0666714 | 0.0736160 |
| Legacy vs native SSM-output control | 0.00616784 | 0.00579150 |

This confirms the SSM-output route is the main measured source of the first
hidden activation delta at layer 0. It does not restore the full semantic
state: native SSM output still differs from legacy at position 513 by
`0.0196199` recurrent state, `0.261386` convolution history, `0.25` active K,
and `0.3125` active V (max abs). The layer-0 recurrent/convolution mismatch
remains `0.000592232` / `0.00115204`, identifying the separate Mx QKV path.
CPU-logit cosine is `0.999284796`, below the diagnostic `0.9995` floor, which
is not a formal P512 acceptance threshold. No exact cross-route semantic
equivalence is established.

The native representation used the same 2,637 device allocations as the Mx
interactive capture, but live allocation increased from `18,472,649,108` to
`18,751,832,468` bytes (`+279,183,360` bytes); the selected representation is
larger even though no duplicate SSM-output copy is kept. Decode timings from
these capture runs include host downloads and are not performance evidence.
Raw outputs, state/logit comparisons, telemetry, and captures use the
`*-ssm-native-layerpath*` artifact prefix in the external artifact directory.
This control is diagnostic only and authorizes no performance candidate.

### Current-build graph/direct boundary checks

The existing `--check-graph-state` checker was run on Release binary SHA256
`10b2ddb840123159defec1f004be72a4f29017f7c7260d50b6f5e9fcbd0f0f1e` with
the Qwen3.8-27B-Q4_K_M model (SHA256
`7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`). The
input was `/tmp/m27-context-smoke/p512.txt` (SHA256
`457045510e75c2ca1a3a3a73e3645ca4582e395279dbdcea67bd6495d4bb9afb`),
which tokenizes to 512 inputs and is a repeated quick-brown-fox fixture, not the
varied EXP-0351 prompt used for the canonical cross-route A/B state.

| Route | Context transition | Result |
|---|---|---|
| legacy | 512 -> 513 | PASS, graph/direct tokens and state exact |
| legacy | 2048 -> 2049 | PASS, graph/direct tokens and state exact |
| legacy | 8192 -> 8193 | INTERRUPTED before checker result; diagnostic only |
| interactive | 512 -> 513 | PASS, graph/direct tokens and state exact |
| interactive | 2048 -> 2049 | Unsupported: `m25_interactive` sets cache capacity below 2048; checker rejected the requested context |
| interactive | 8192 -> 8193 | Unsupported for the same selector capacity limit |

The 8192 legacy attempt spent several minutes prefilling; telemetry recorded a
96 C junction temperature and one SCLK sample at 1485 MHz. It was interrupted
before correctness output and cannot count as a pass. The other successful
checks are same-route graph/direct parity only; they do not establish
legacy-versus-interactive state equivalence or accepted cross-route numerical
tolerance. Raw logs and the continuous telemetry file are in
`/home/fedora-workstation/Development/mi50-artifacts/exp0358-current-route/20260920/boundary-current-build/`.

At the time of this boundary probe, the current cross-route TG128 correctness
gate remained open and no matched timing was started. M26-CQ later closed the
semantic comparison as `ROUTES_NOT_COMPARABLE`; see the final re-evaluation
below.

### Combined recurrent selector control

To test whether the separately identified QKV and SSM-output sources explain
the first-step activation drift together, the current M26-C selectors
`--m26c-recurrent-qkv-m23` and `--m26c-ssm-out-native` were enabled together
for one direct TG1 decode from the exact `p512-varied.state` snapshot. Release
binary SHA256 remained
`10b2ddb840123159defec1f004be72a4f29017f7c7260d50b6f5e9fcbd0f0f1e`; model
hash is the Qwen3.8-27B hash recorded above. The run emitted token `1406`,
matching legacy. Continuous telemetry contained 577 samples, all SCLK/MCLK
`1606/1000 MHz`, with junction temperature 33–36 C.

Against the current-build legacy capture, layer-0 QKV max abs was
`1.90735e-5`, recurrent output `4.76837e-7`, gate projection `0.0243430`,
gated activation `0.000654325`, post-SSM-output residual `0.00616911`, and
layer output `0.00580125`. Thus the combined selector removes most of the
first-layer output error associated with the default interactive SSM-output
path, while the gate projection delta remains. Full semantic state is still
not equivalent at position 513: recurrent max abs `0.0177761`, convolution
history `0.229127`, active K `0.216309`, and active V `0.489258`. The CPU
reference logit comparison reports argmax `1406` and cosine `0.999820632`; this
is a single-position diagnostic, not a TG128 criterion or an accepted
cross-route tolerance.

The capture decode duration includes diagnostic downloads and is not a timing
sample. Raw state, logits, layer tensors, logs, and telemetry are retained as
`qkv-ssm-native-combo*` in the external artifact directory. The combined
selector narrows the first-layer activation attribution but does not establish
cross-route equivalence, so A-D timing remains gated and no optimization is
authorized.

### Combined-selector TG128 correctness result

A fresh legacy TG128 and the combined `recurrent-qkv-m23` + `ssm-out-native`
interactive TG128 were run from the same `p512-varied.state` using the same
current Release binary SHA256
`10b2ddb840123159defec1f004be72a4f29017f7c7260d50b6f5e9fcbd0f0f1e` and the
same model hash. Both ran direct decode for 128 tokens; their generated token
ID arrays compare exactly. Both reach position 640 with pending token `21`.
Continuous telemetry for both runs recorded SCLK/MCLK `1606/1000 MHz` (junction
33–36 C).

Canonical semantic state still fails the existing exact comparison at that
frontier: recurrent max abs `0.726309`, convolution history `2.883845`, active
K `8.205078`, and active V `34.28125`. The comparison log records the full
per-layer differences. Exact token parity for this one TG128 probe does not
supply an accepted state tolerance or establish equivalence for a following
decode step. These combined diagnostic selectors also do not represent the
unmodified interactive route in A-D. Their single-run decode times are not a
matched performance result. At the time, the state-equivalence gate remained
open; M26-CQ later closed the route comparison as non-comparable under an
accepted numerical contract.

Artifacts are `legacy-current-tg128*`, `qkv-ssm-native-combo-tg128*`, and
`current-legacy-vs-qkv-ssm-native-combo-tg128.compare.log` in the external
artifact directory. No candidate or timing qualification is authorized.

### Current-binary principal-route TG128 gate

To remove the earlier build-identity gap, unmodified legacy and unmodified
`m25_interactive` direct decode were each run from the exact same
`p512-varied.state` on Release binary SHA256
`10b2ddb840123159defec1f004be72a4f29017f7c7260d50b6f5e9fcbd0f0f1e`, with
the same Qwen3.8-27B model hash. Both emitted token `1406` first. Current
legacy and interactive telemetry recorded SCLK/MCLK `1606/1000 MHz`; the
interactive run had 414 samples and 32–55 C junction temperature. The TG128
state comparison fails: generated IDs first differ at index 55 (legacy `16`,
interactive `15`), processed histories first differ at absolute index 568,
and final pending tokens at position 640 are `21` and `17`. Final semantic
state maxima are recurrent `2.871509`, convolution history `20.115730`, active
K `20.744141`, and active V `103.960938`.

The common semantic frontier is therefore the restored position-512 state
with pending input token `2193`, before either route consumes token 513. The
routes do not preserve state equivalence after decode: the legacy combined
native recurrent projection path and the interactive resident Mx path produce
different QKV/gate/SSM-output values, then diverge in hidden activations, state,
and ultimately token IDs. The TG128 timings printed in the raw logs are
excluded because the sequences cease to match at token index 55 and are not a
matched steady-state comparison. M9 provides a CPU-oracle logits contract at
teacher-forced position 64, but no accepted pairwise cross-route tolerance for
recurrent state, convolution history, or KV. The M26-C state-equivalence gate
was not met; do not run A-D qualification or calculate route-delta shares from
these incomparable runs. M26-CQ later closed the semantic investigation as
`ROUTES_NOT_COMPARABLE` under the available numerical contracts; see the final
re-evaluation below.

At the time of these measurement-path results, the correctness gate remained
open. M26-CQ later closed the route comparison as `ROUTES_NOT_COMPARABLE`
under the available numerical contracts; see the final re-evaluation below.
Do not start an optimization or A-D timing qualification. Artifacts:
`legacy-current-tg128*`,
`interactive-current-tg128*`, and `current-legacy-vs-interactive-tg128.compare.log`
in the external artifact directory.

### Source-level selector trace at the first divergence

The P512 snapshot import reports the same persistent representation for both
principal routes: `gdn-state-transposed|kv-f16`. Snapshot restore therefore
reaches the same semantic frontier and converts into the same declared
recurrent-state/KV representation. The first observed divergence is in decode
projection values, not a different serialized state layout.

| Family | Legacy current selector/path | Interactive current selector/path | Evidence / implication |
|---|---|---|---|
| Recurrent QKV | Combined native Q4_K wave GEMV over QKV+gate input | Resident Mx Q8 MMQ for QKV | Layer-0 QKV max abs `0.00115204`; M23 diagnostic control reduces it to `1.90735e-5` |
| Recurrent gate | Shares the legacy combined native QKV+gate projection | Separate resident Mx Q8 MMQ projection | Gate projection remains `0.0243430` different with QKV-M23 and native SSM-output controls |
| Recurrent SSM output | Native Q5_K wave GEMV | Resident Mx Q5_K MMQ | Native control lowers layer-0 post-projection residual error from `0.0666714` to `0.00616784` |
| Recurrent update | Fused recurrent core with transposed state | Same fused recurrent core with transposed state | Layout/kernel family matches; its Q/K/V inputs are already numerically different |
| Recurrent FFN | Native fused Q4_K Gate/Up/SwiGLU and native Down paths | Separate Mx Gate and Up projections, SwiGLU, then Mx Down | Layer-0 FFN output differs even after input residuals are close; these families remain changed by recurrent-only controls |

These branches are in `RecurrentLayer::step()` and its constructor in
`tools/qwen35_gpu_pipeline.hpp` (current source around lines 1070–1320,
3020–3450). The diagnostic QKV-M23 + native-SSM combination leaves the gate
and FFN on Mx and only substitutes those two selected projections. It reaches
exact TG128 token parity against current legacy, but final recurrent and KV
state remain different; it does not establish an accepted semantic tolerance.

This trace explains why representation normalization alone does not make the
routes comparable: the common state layout still receives route-specific
quantization and matrix paths. Route-level dispatch and synchronization costs
remain unqualified because correctness fails before a matched timing boundary.

### Canonical snapshot field classification

The current M26-C snapshot schema (`M26CSTAT`, version 1) classifies as
follows. It captures the minimum model state at position 512 and intentionally
does not serialize route workspaces or graph objects.

| Category | Fields | Representation / handling |
|---|---|---|
| A. Semantic model state | Pending current input token; absolute position; processed token history; logical recurrent state for every recurrent layer; convolution history; active K/V for each attention head | Recurrent and convolution tensors serialize as FP32 logical values. Active K/V serializes as canonical FP32; active length is `position` entries per head. Position equals processed-history length and determines the next cache slot/valid length. |
| B. Route-specific persistent representation | Source runtime contract, selectors, recurrent-state layout, KV dtype/layout; target engine’s corresponding state layout | Source identity is recorded for interpretation and diagnostics. Restore converts the canonical logical state to the target recurrent layout and target KV dtype. These are not duplicated as second semantic state copies. Model SHA256, tensor quantization identity, architecture dimensions, and model context length are checked. |
| C. Temporary scratch/workspace | Q8/Q8_1 staging buffers, normalized activations, QKV/gate/FFN temporaries, recurrent outputs, projection buffers, logits/sampling scratch, allocator workspace | Excluded; each route allocates/reuses its own scratch after engine construction. |
| D. Graph/runtime metadata | HIP graph handles, captured node topology, replay counters, stream/event handles, timers, telemetry, host capture buffers | Excluded; each route constructs its own runtime/graph metadata. Generated output token IDs are retained in the snapshot only as diagnostic history; import validates but does not restore them as model state. |

The route-independent cursor is represented by `position`, `current_token`,
and `history.size()==position`; the active KV prefix uses that same position.
The source contract string reports preset/planner choices, recurrent layout,
KV type, and sorted `MIINFER_*` selectors. Binary/ROCm/GPU hashes are kept in
the run records and experiment metadata rather than embedded in schema v1.
This distinction is why equality is judged on the restored semantic fields,
not on route-contract strings or scratch bytes.

### Final re-evaluation — M26-CQ semantic-equivalence investigation

EXP-0359 teacher-forced both unmodified direct routes with the same 68-token
sequence at absolute positions 512–579 from the common varied-text P512 state.
The processed token histories and snapshot positions match exactly at all 12
checkpoints; recurrent, convolution-history, and active KV fields have the
same dimensions and logical layouts. Both remain finite. Numeric state drift
is present after the first input and grows through the checked sequence. At
input position 567, shared input token `13` produces top-1 `16` for legacy and
`15` for interactive; the top-1/top-2 margins are `0.0099449` and `0.0083122`.
The layer trace shows exact layer-0 input/normalized input, then route-specific
QKV/gate values from the first decode input; this predates the greedy flip.

The M9 CPU-oracle gate is tied to its position-64 fixture and does not establish
a pairwise tolerance for P512 recurrent state, convolution history, KV, final
norm, or logits. No shared accepted cross-route numerical contract applies to
these measurements. Therefore the route-differential question is closed as
**`ROUTES_NOT_COMPARABLE`**. No semantic fix or performance change was made;
the interactive route remains experimental and no A–D timing qualification is
valid. Retain the qualified no-preset path as canonical and plan subsequent
M26-E/M26-F work using that path. Full evidence and exact commands are in
[EXP-0359](EXP-0359-m26cq-decode-semantic-equivalence.md) and raw artifacts at
`/home/fedora-workstation/Development/mi50-artifacts/exp0358-current-route/20260921/m26cq/`.
