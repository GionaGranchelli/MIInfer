# M30-0002 production authority and residual agent cost

Status: complete; the required normal-reuse 40-turn/86-request residual run
completed on the Machinist MI50 and all telemetry rows are present.

## Decision

`PREFILL_V2_IS_AUTHORITY; QWEN35_RUNTIME_RETAINED_AS_COMPATIBILITY_PATH`.

The current `miinfer serve` path constructs `PrefillV2Model` in
`tools/miinfer_cli.cpp:5048`, enables prefix reuse in its `GenerateOptions`,
and maps `GenerateStats` into the request telemetry. `Qwen35RuntimeEngine` is
constructed at `tools/miinfer_cli.cpp:4236` by a different CLI path. This is
therefore case B: two materially separate execution architectures, with the
divergence at the CLI dispatch/engine-construction boundary. No serving
migration is required for this decision.

## Current production path

```text
miinfer serve
  -> cmd_serve
  -> Qwen35Model + tokenizer
  -> PrefillV2Model
       -> GenerateOptions(enable_prefix_reuse, cache_prefix_after)
       -> ReusableContext (one in-memory GDN checkpoint + final hidden)
       -> 48 recurrent/GDN state stores
       -> 16 attention KV stores
            -> ContextSpace -> PlacementPlan -> DeviceKvPool
            -> resolved PhysicalKvView -> attention kernels
       -> suffix-only prefill or cold 512-token macro tiles
       -> persistent recurrent/KV state -> single-token decode
```

`PrefillV2Model` owns model execution, prefill, decode, recurrent state, and
KV state. `AttentionLayerKvCacheStorage` resolves the physical view during
construction; the hot attention path consumes raw pointers and layout metadata.
The M29 qualification record identifies this same ContextSpace/PlacementPlan/
DeviceKvPool/PhysicalKvView chain as the shipped production N=1 path and
records the 128K envelope at about 31.07 GiB static VRAM.

`Qwen35RuntimeEngine` is retained in the CLI for the other run/benchmark route.
It owns a separate layer graph, wide-prefill workspace, direct recurrent/KV
buffers, and an independent eight-checkpoint, 3-GiB session cache. It does not
construct or call the Prefill V2 `ContextSpace`, `PlacementPlan`,
`DeviceKvPool`, or `PhysicalKvView` path. Its richer multi-checkpoint behavior
must not be mistaken for serving behavior.

| Responsibility | `PrefillV2Model` / `serve` | `Qwen35RuntimeEngine` / other CLI path |
|---|---|---|
| Model execution | `PrefillV2Model::forward`, `generate`, and `decode_step` | `generate_layer_major` / legacy layer objects |
| Prefill | 512-token full-model macro tiles; suffix-only continuation on a hit | Wide/layer-major runtime scheduler |
| Decode | Persistent V2 recurrent/KV state handed directly to single-token decode | Runtime engine's separate decode graph and buffers |
| KV ownership | 16 `AttentionLayerKvCacheStorage` instances with resolved `PhysicalKvView` | Direct runtime-layer key/value buffers |
| GDN ownership | 48 `RecurrentLayerStateStorage` instances plus `ReusableContext` checkpoint | Runtime-layer recurrent buffers plus `SessionCheckpoint` copies |
| Exact-prefix reuse | One active in-memory exact checkpoint; persistent files are optional | Longest-prefix radix index over up to eight in-memory checkpoints |
| Zero suffix | Supported with the in-memory final-hidden payload | Separate runtime checkpoint semantics; not the serving path |
| Context/placement | `ContextSpace` -> `PlacementPlan` -> `DeviceKvPool` -> `PhysicalKvView` | No calls to those M29 placement types in this class |
| 128K | Current server accepts 131072; M29 records a focused 128K envelope | Runtime has a configurable cache capacity, but is not the M29 shipped authority |
| Integration | `cmd_serve` constructs it and maps its telemetry | `cmd_run`/benchmark-style CLI path constructs it |
| M29 relationship | The qualified physical-view production owner | Retained compatibility/measurement architecture |

## Reuse semantics audit

| Capability | Current serving behavior |
|---|---|
| Active in-memory checkpoints | One; `ReusableContext` owns one token vector, GDN checkpoint, and optional final hidden vector. |
| Match rule | Exact token prefix plus model, quantization, and state-layout identity. |
| Longest-prefix match | No for the in-memory checkpoint; yes across persistent `.miinfer` files, where the longest matching file is selected. |
| Suffix continuation | Yes; restore GDN state and existing GQA KV, then execute only the suffix. |
| Zero-suffix continuation | Yes for an in-memory checkpoint with final hidden; persistent sessions lack that payload and fall back to a correct cold path. |
| Multiple retained checkpoints | Not in the in-memory serving cache. Persistent disk sessions can retain multiple files. |
| Shared prefixes/refcount/COW/fork/rollback/Tail-Replay | No implementation in the serving path. |
| Invalidation | Token/model/quantization/layout mismatch prevents reuse; `ReusableContext.clear()` invalidates the in-memory checkpoint. Reset clears active state; cancellation/failure handling is not a multi-checkpoint serving mechanism. |
| State cost | One GDN checkpoint is 158,859,264 bytes; KV remains in the model's 16 persistent attention stores. Persistent disk sessions additionally serialize GDN and KV bytes. |

## Residual canonical workload

The fixed M30 transcript was replayed once with normal production reuse
enabled on the Machinist MI50. The run used the current release build, model
`/home/machinist/models/Qwen3.8-27B-Q4_K_M.gguf`,
`HIP_VISIBLE_DEVICES=1`, `--context 131072`, and the existing command:

```text
python3 bench/m30_fixed_transcript_replay.py \
  --baseline results/m30-0000-agent-workload-corrected.json \
  --url http://127.0.0.1:8090/v1/chat/completions \
  --output results/m30-0002-residual.json
```

The artifact contains 40 turns/86 requests, passes the all-turns minimum
generation gate, and matches 86 server latency rows and 86 request rows. No
cold 86-request comparison was required.

| Cost family | Measured result | Avoidable? | Interpretation |
|---|---:|---|---|
| Suffix prefill | 1,572,308 tokens; 681,661.874 ms on reuse hits | mostly no | Existing exact-prefix reuse already executes only the suffix. |
| Residual prefix replay | 1,516,386 tokens across 33 cold/fallback requests | yes if a valid prior checkpoint exists | This is the largest remaining avoidable token category. |
| State restore/update | restore 41.995 ms total; save/update not separately instrumented | restore is negligible; update unknown | Restore is not the wall-time bottleneck; capture/update is included in existing timing or the residual remainder. |
| Branch/retry replay | no distinct branch/retry marker; cold rows are the evidence available | workload-dependent | The run does not prove that every cold row is a retry or branch. |
| Decode | 2,271,402.44 ms total | mostly no | 17.3473% of measured request wall time; current decode path is already production behavior. |
| Runtime/host overhead | 4,454.492 ms after prefill+decode | low | 0.034% of measured request wall time; tokenization was 4,139.725 ms and queue wait 7.388 ms. |

Aggregate request wall time was 13,093,710.62 ms. Prefill accounted for
10,817,853.688 ms (82.6187%), decode for 2,271,402.44 ms, and the measured
remainder for 4,454.492 ms. Logical prompt volume was 3,626,268 tokens;
2,053,960 (56.641%) were reused and 1,572,308 (43.359%) were executed as
suffix/new prefill. The reuse count was 53 hits versus 33 cold/fallback rows.

The serving checkpoint remained one active in-memory checkpoint, with a
reported reuse checkpoint size of 8,748,793,856 bytes. Startup telemetry
reported 34,342,961,152 total device-VRAM bytes and 117,440,512 free bytes;
there is no per-request VRAM peak series in this server log.

## Final capability recommendation

The single highest-value missing capability indicated by this run is a
serving-side multi-checkpoint/longest-prefix retention mechanism for genuine
branch or retry workloads. The evidence is the 1,516,386-token residual replay
on 33 cold/fallback requests while the current in-memory serving authority
retains only one checkpoint. This is a recommendation for the next capability,
not an implementation claim: this transcript does not independently prove that
all 33 cold rows represent branches or retries, so a follow-up workload should
first validate that causal pattern. No new suffix, restore, decode, or host
overhead optimization is justified by these measurements.

## Completion gates

| Gate | Result |
|---|---|
| Serving execution path identified | PASS — `miinfer serve` constructs `PrefillV2Model`. |
| M29 relationship understood | PASS — the qualified chain is ContextSpace -> PlacementPlan -> DeviceKvPool -> PhysicalKvView. |
| Existing reuse semantics documented | PASS — one active exact-prefix checkpoint, suffix continuation, zero-suffix behavior, invalidation, and persistent-session distinction are recorded above. |
| Residual agent wall-time attributed | PASS — prefill, suffix prefill, restore, decode, tokenization, queue wait, and residual remainder are measured. |
| Remaining prefix replay quantified | PASS — 1,516,386 replayed tokens across 33 cold/fallback rows. |
| Next capability selected from evidence | PASS with qualification — multi-checkpoint/longest-prefix retention is selected, pending causal validation with a branch/retry workload. |

## Historical boundary

M30-0001's original premise was to introduce exact-prefix hybrid-state reuse.
The source and bounded production A/B show that equivalent production reuse was
already present. M30-0001 correctness and zero-suffix tests pass; the bounded
reuse-on/reuse-off result is 64.8% fewer physical prompt tokens and 43.1% lower
total wall time. The full canonical cold replay was deliberately stopped at
14/86 because the bounded A/B had already established the production benefit.
Those results do not claim that M30-0001 caused the existing advantage.

## Completion fields

```text
M30_0002_BASE_SHA=2aecf03
RUNTIME_AUTHORITY_RESULT=PREFILL_V2_IS_AUTHORITY_QWEN35_RUNTIME_COMPATIBILITY
RESIDUAL_ANALYSIS_SHA=d3f3866fd6f18b0d4e65909419c16866f9812e36
SEMANTICS_AUDIT_SHA=9644c6d
INTEGRATED_EVIDENCE_SHA=4adbbd692b0200d2e0033d7f349fd35b44372744
```
