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
M31_0002_BENCHMARK_SHA=3c07a65138b3a823fc1cd21cc0c168ba372878fa
M31_0002_RESULT=FAIL
```

The final card-1 diagnostic run used the frozen model and ROCm 7.2.1
container. The integrated flow reached cached-prefix reuse, but greedy output
diverged from the cold replay at generated token 11 (`expected=3299`,
`actual=4434`; both sequences contained 32 tokens). The exact artifact is
`results-m31-0002-card1-final-fix.txt` (SHA-256
`8e5dd2ceef1ce1b29ee3644d146b4faccb678b58d87776480523027024741fd7`). The
qualification therefore remains failed; snapshot/COW metrics are not promoted
as a passing integrated result. A prefill-only cache-priming path and
synchronous null-stream GDN copies were added and rechecked; the same
deterministic mismatch remains, so the root cause is still an unresolved
in-memory prefix-reuse state/parity issue rather than the original one-token
priming mutation.
