# M31-0002 — Persistent-Context Qualification

## Scope

Qualify the integrated M29/M30 persistent-context path on the frozen N=1
runtime. This stage combines the already-proven prefix reuse, snapshot,
fork/rollback, nested branch, and COW mechanisms into one measured flow. It is
not a COW optimization task.

## Required flow

```text
cold context
  -> warm same-context continuation
  -> cached-prefix suffix continuation
  -> canonical snapshot
  -> sibling fork
  -> divergent COW mutation
  -> nested fork and rollback
  -> resume canonical branch
```

Every branch must be compared with a cold replay for output parity. The test
must also exercise invalid/released snapshot rejection and final cleanup.

## Required metrics

```text
replayed_tokens
avoided_tokens
cold_ttft_ms
warm_ttft_ms
reuse_ttft_ms
restore_ms
fork_ms
checkpoint_bytes
peak_vram_bytes
output_parity
```

The evidence must include COW shared/private bytes, reference count, COW event
count, and COW bytes copied where the implementation exposes them.

## Evidence

```text
M31_BASELINE_SHA=99840d10eabe5e64fe2564d34d3d3fcd42b43521
M31_0002_BENCHMARK_SHA=<filled after implementation>
M31_0002_RESULT=<PASS or FAIL>
```
