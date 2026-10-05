# M30-0002 production authority and residual agent cost

Status: analysis in progress; the required normal-reuse 40-turn/86-request
residual run is active on the Machinist MI50.

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

The fixed M30 transcript is being replayed once with normal production reuse
enabled on the Machinist MI50. The final report will aggregate the server's
per-request telemetry into this table; no cold 86-request replay is required.

| Cost family | Session cost | Avoidable? | Existing mechanism | Candidate next mechanism |
|---|---:|---|---|---|
| Suffix prefill | pending | mostly no | exact-prefix suffix execution | — |
| Residual prefix replay | pending | yes if present | exact-prefix match/fallback | measure before selecting |
| State restore/update | pending | maybe | GDN restore and checkpoint capture | measure before selecting |
| Branch/retry replay | pending | workload-dependent | single in-memory checkpoint | measure before selecting |
| Decode | pending | mostly no | current decode kernels | later |
| Runtime/host overhead | pending | maybe | sequential server loop | measure before selecting |

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
RESIDUAL_ANALYSIS_SHA=pending
SEMANTICS_AUDIT_SHA=pending
INTEGRATED_EVIDENCE_SHA=pending
```

