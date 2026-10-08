# M31-0003 — Consolidated findings and bounded next steps

## Scope and identity

Source baseline: `HEAD dc7370b76e7f3fdb01c2c810cd07181703bef2d2` plus the
pre-existing dirty tree and this task's targeted edits. This report traces the
current working tree, not a clean commit. No GPU inference, profiling,
temperature experiment, or new benchmark was run. The user reports both
machines are physically in good condition; hardware/cooling diagnosis is not
part of this source-only task. A high temperature alone does not prove
redundant computation or a code defect.

## Consolidated ranking

| Component | Finding | Class | Potential impact | Risk | Priority |
|---|---|---|---|---|---:|
| GQA/KV | Causal prefill attention is O(N²); decode scans KV proportional to context. At 128K, one logical read per shared KV head is ~8 GiB/token across layers. | Confirmed algorithmically; traffic is a model | High at long context if a safe kernel reduces repeated work | High: cache reuse, numerical order, occupancy; earlier LDS-sharing trial regressed | 1 (measure then design) |
| Prefill/GDN | GDN is ~66% of one 8K profile; Q/K load/work fans out to 3 value heads per key head. | Confirmed at 8K/source; 32/64K cause unknown | Potentially high, linear in tokens/layers | High state-parity and register-pressure risk | 2 (32/64K attribution) |
| Dispatch | Successful cached prefill repeatedly copies the full 151.5 MiB checkpoint at each 512 boundary; up to ~40.67 GB/24,832 HIP copy calls at 128K. | Confirmed from current source; runtime impact unmeasured | Potentially meaningful host/API and copy overhead, but likely bounded vs whole inference | Moderate: intermediate checkpoint/cancellation behavior | 3 (separate behavior change) |
| Decode/MMQ | `generate()` replays one-token graph then copies full logits D2H/synchronizes each token for CPU sampling. | Confirmed source; timing share unmeasured | Constant per output token | Medium/high if sampling semantics change | 4 (measure decode path first) |
| Memory | One-shot model eagerly allocated 158,859,264-byte GDN checkpoint despite no prefix cache. Lazy allocation is implemented. | Confirmed and corrected | ~151.5 MiB less reserved when unused | Low; capture/restore unchanged | Done |

## What is established—and what is not

The production `run`, `chat`, and `serve` paths use `PrefillV2Model::generate`.
Cold prefill, graph-backed suffix prefill and one-token decode take different
routes. Kernel and timing evidence available for long-context work is not
component-attributed at 32K/64K: the 32K guarded profile ended at its thermal
guard before emitting a row, and 64K profiling was not run. The historical
8K profile and directional llama.cpp comparisons must not be stretched into a
128K root-cause claim.

High-confidence code opportunities are not all safe to roll out immediately.
Repeated checkpoint copies are unnecessary to retain the final state on a
successful request, but they may represent intermediate-prefix availability;
failure/cancellation semantics and short-prompt reuse need a dedicated test
before removing them. GQA/GDN sharing and changing decode sampling are larger,
GPU-dependent kernel/runtime projects. The already-rejected EXP-0360 sharing
kernel and non-comparable EXP-0358 route are not reopened.

## Implemented fix

`GdnCheckpointStorage` now allocates its large state/history buffers on first
capture instead of model construction. No-prefix-cache paths save 158,859,264
bytes of reserved VRAM; prefix caching still allocates the same capacity and
uses the same copies. The new mocked-HIP host test verifies allocation timing,
accounting, copy count, valid capture and clear semantics. It does not establish
HIP/MI50 runtime behavior or temperature change.

## Dependency-aware next optimization sequence

1. Keep the current source/data findings frozen; do not begin kernel tuning
   based on the single 8K profile.
2. In a separately authorized, short, guarded run, obtain matched 8K and then
   32K component attribution with per-call flushed results and telemetry. Stop
   before escalation if thermal limits are approached. 64K only follows a
   diagnostic that survives.
3. Measure checkpoint-save time/call count with prefix caching enabled and
   test exact-prefix, suffix, short prompt, and failure/cancel behavior. Then
   decide whether one final save can replace every 512-boundary save.
4. Only after attribution, prototype one GDN Q/K-sharing or GQA-sharing
   candidate; require existing parity checks and a small targeted component
   comparison before any longer context.
5. Measure ordinary decode's logits round trip separately before moving
   sampling onto the GPU or changing graph orchestration.

No performance gain is claimed. Do not start 64K/128K matrices or another
milestone automatically.

## Offline validation and outstanding GPU work

The mocked-HIP allocation test passed. Existing host-only tests and static
source checks are separate from GPU validation. Still required on real MI50:
confirm allocation/reuse parity, check actual VRAM delta, capture 32K/64K
component timing under safe telemetry, determine whether checkpoint copies
matter to latency, and verify any later kernel candidate. The active M31-0002
32K/64K attribution gate remains open.
