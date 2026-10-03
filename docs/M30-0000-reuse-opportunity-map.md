# M30-0000 — Reuse Opportunity Map (Workstream B)

## Exact status

`M30_REUSE_OPPORTUNITY_MAPPED`

This is the Machinist-only Workstream B source/measurement record. It does not
claim the coordinator's end-to-end agent baseline: the canonical workload was
not frozen or executed on this host, and no reuse mechanism was implemented.

| Identity | Value |
|---|---|
| `M30_0000_BASE_SHA` | `23deab2eda689fc0cfc21782d3f6df1303bee2ee` |
| branch | `m30/reuse-opportunity-map` |
| host | `machinist@100.114.213.94` |
| device selection | `HIP_VISIBLE_DEVICES=1` |
| selected device | `gfx906:sramecc-:xnack-`, `34,342,961,152` bytes VRAM |
| model | `Qwen3.8-27B-Q4_K_M`, 64 layers |
| workload version | `UNFROZEN_COORDINATOR_INPUT` |

The existing HIP smoke test reported the selected device. Existing bounded
tests also passed: `ContextSpace host tests`, `PhysicalKvView single-shard
resolution`, and `DeviceKvPool physical test`.

## Production state map

The model is 16 topology blocks, each with three recurrent layers and one GQA
layer: `include/miinfer/prefill_v2/constants.hpp:34-36` and
`src/prefill_v2/model.cpp:208-224`. The model therefore owns 48 recurrent
states and 16 KV caches (`include/miinfer/prefill_v2/model.hpp:77-84`).

### GQA KV state

`AttentionLayerKvCacheStorage` allocates one placement-owned physical block
and resolves a `PhysicalKvView` once (`src/prefill_v2/kv_cache.cpp:10-89`).
The default FP16 layout is `[KV-heads, capacity, head-dim]` for both K and V
(`include/miinfer/prefill_v2/kv_cache.hpp:43-59,100-119`). For this model:

```text
bytes/token = 16 GQA layers * 2(K,V) * 4 KV heads * 256 head_dim * 2 FP16
            = 65,536 bytes = 64 KiB
```

The store kernel writes the new range at
`[base_position, base_position + token_count)` and receives the cache capacity
(`src/prefill_v2/attention_layer.cpp:296-305`). Decode writes one token at
`position` (`src/prefill_v2/attention_layer.cpp:598-609`). Thus a valid append
requires the incoming prefix to remain unchanged and the next position to be
the previous coverage end; there is no logical-page traversal in the kernel
view.

### GDN/recurrent state

Each recurrent layer owns a state matrix `[48,128,128]`, a four-token
convolution history `[4,10240]`, and a position scalar
(`include/miinfer/prefill_v2/state.hpp:16-24`). The state is allocated and
zeroed by `RecurrentLayerStateStorage` (`src/prefill_v2/state.cpp:10-14,43-51`).
Forward copies incoming state when buffers differ, runs convolution and the
GDN scan in place, then sets `position = base_position + token_count`
(`src/prefill_v2/recurrent_layer.cpp:281-341,391-397`). Single-token decode
updates the same convolution history and recurrent matrix
(`src/prefill_v2/recurrent_layer.cpp:557-616`).

There are 48 such states. Exact fixed storage is:

```text
state/layer   = 48*128*128*4 + 4*10240*4 = 3,309,568 bytes
all 48 layers = 48*3,309,568 = 158,859,264 bytes = 151.50 MiB
```

The convolution history is part of the resume state; a recurrent matrix alone
is insufficient.

### Request lifecycle and coverage

`prefill_sequence` currently starts at position zero and dispatches every
prompt token in 512-token macro-tiles (`src/prefill_v2/model.cpp:587-627`).
The normal generation path resets state and then dispatches the complete prompt
(`src/prefill_v2/model.cpp:852-873`). Therefore, for turn N+1 with an exact
common prefix of P tokens and a changed suffix of S tokens, the current cold
path physically processes `P+S` tokens even though only S are new to the
application. GDN state and GQA KV coverage both advance through the whole
prompt.

The final hidden activation used to produce first-token logits is not part of
the GDN checkpoint; the existing reuse code explicitly documents this boundary
(`src/prefill_v2/model.cpp:764-773`). An exact-prefix request with zero suffix
therefore needs special handling rather than blindly resuming.

## Invalidation and ownership conditions

The existing fingerprint checks model ID, quantization, state-layout version,
prompt length, and every cached prefix token
(`src/prefill_v2/reusable_context.cpp:155-184`). It rejects empty/invalid
state, version mismatch, model mismatch, quantization mismatch, shorter
prompts, and token mismatch. `reset_state` clears all 48 recurrent states and
16 KV caches (`src/prefill_v2/model.cpp:400-407`).

The source-supported safe reuse condition is therefore:

```text
same model + same quantization + same state layout + exact token prefix
+ prefix length <= new prompt length + same live physical cache ownership
```

Changing the model/configuration/token stream or discarding the live physical
KV owner invalidates the state. These are ownership/layout consequences of the
source path, not new invalidation rules.

## State-size accounting

The following values are exact FP16-GQA plus fixed-FP32-GDN derivations from
the layouts above. `cached_tokens_` adds `4*P` bytes to an in-process
fingerprint; persistent serialization also has a 168-byte header, as recorded
by `include/miinfer/prefill_v2/persistent_session.hpp:19-33`.

| retained context P | GQA KV | GDN state+history | GQA+GDN | cached token IDs |
|---:|---:|---:|---:|---:|
| 8K | 536,870,912 B / 512.0 MiB | 158,859,264 B / 151.5 MiB | 695,730,176 B / 663.5 MiB | 32,768 B |
| 32K | 2,147,483,648 B / 2,048.0 MiB | 158,859,264 B / 151.5 MiB | 2,306,342,912 B / 2,199.5 MiB | 131,072 B |
| 64K | 4,294,967,296 B / 4,096.0 MiB | 158,859,264 B / 151.5 MiB | 4,453,826,560 B / 4,247.5 MiB | 262,144 B |
| 128K | 8,589,934,592 B / 8,192.0 MiB | 158,859,264 B / 151.5 MiB | 8,748,793,856 B / 8,343.5 MiB | 524,288 B |

The GQA value is retained in the live 16-cache model for in-process reuse;
the 128K row is an accounting result, not a new 128K allocation or runtime
qualification on this branch.

## Historical evidence (not current qualification)

- `experiments/EXP-V2-0014-prefix-state-reuse.md` reports historical exact
  prefix/suffix experiments: at 64K+512, cold TTFT 1,282,069.65 ms versus
  reuse TTFT 17,406.08 ms, with a 151.50 MiB GDN checkpoint. It is prior
  evidence, not a Workstream B run at `23deab2`.
- `experiments/EXP-V2-0028-persistent-prefix-cache.md` reports historical
  cross-process serialization at P=4096: 407.52 MiB total payload and 491.2
  ms restore. It retains both GDN and GQA state, unlike the in-process GDN
  checkpoint alone.
- `experiments/V2-0048A-generation-budget-and-exact-prefix-fault.md` records a
  historical zero-suffix exact-prefix aperture fault caused by using an
  uninitialized final hidden location. This is a blocking correctness edge
  for any future reuse enablement, not a fix made here.

## Opportunity map

| Mechanism | Repeated work today | Potential work avoided | State cost | Dependency | Evidence |
|---|---:|---:|---:|---|---|
| Exact-prefix KV reuse | P GQA tokens reprocessed on cold path | GQA attention over P | 64 KiB/P token | exact token/model/layout match; live physical owner | KV store and cold lifecycle refs above |
| GDN recurrent checkpoint reuse | P tokens re-run through 48 GDN layers | GDN prefix scan/convolution | 151.50 MiB fixed | matrix **and** conv history at P | `state.hpp`, `reusable_context.cpp` |
| Append-only suffix execution | `P+S` dispatched | only S dispatched | GQA linear in P; GDN fixed | contiguous coverage and final-logit boundary | `model.cpp:587-627,852-873` |
| Shared-prefix pages | no measured production sharing | not established | unknown | refcount/COW ownership design absent | no source evidence; do not implement |
| Tail-Replay | no measured production path | not established | unknown | no source evidence | historical/current source search only |

The largest measured historical opportunity is exact-prefix reuse with
suffix-only execution, but this record makes no performance claim for the
current branch. It is the single recommended next capability for M30-0001;
do not implement it as part of M30-0000.

## Blockers and boundaries

1. The coordinator has not supplied a frozen/executed canonical 20–50-turn
   workload, so Workstream B cannot honestly report total session wall-clock,
   logical prompt tokens, or physically processed prompt tokens.
2. Reuse/session source already exists in this candidate from historical
   milestones, but was not changed or newly qualified here. The known exact-
   prefix/zero-suffix fault must be resolved and independently qualified before
   any future enablement.
3. A 128K combined footprint is an accounting result only. Whether a chosen
   deployment can retain it depends on model weights, workspace, and live
   capacity; this branch did not allocate a new 128K session.

No M30 reuse mechanism, M31 feature, or M29 architecture change was added.
