# M31-0004 — Correctness and validation

## Correctness issue fixed

The runtime used `ReusableContext::resident_` to mean that live GDN state
matched the saved prefix. A cold 513-token request could capture token 512,
advance live state by one token, and leave the saved 512 checkpoint marked
resident. A later exact hit could then skip restoring the checkpoint. Similar
after-the-fact invalidation could be bypassed if a callback or HIP operation
threw before the epilogue.

The runtime now invalidates cached residency before prefill and decode entry
points, graph launches that bypass those entry points, snapshot restoration, and
access through mutable state/context accessors. Successful capture/restore
marks the cache resident again. This is conservative: mutable-accessor reads
can require a redundant future restore, but cannot silently preserve a stale
“live state matches checkpoint” claim.

On successful completion, the full prompt is captured even when it is shorter
than 512 tokens or ends on a partial tile; the already-captured final aligned
boundary is not captured a second time.

## Fingerprint extension contract

The general `ReusableContext::save` remains the safe arbitrary-prefix operation
and computes the full FNV-1a hash. `save_extension` requires an existing valid
checkpoint at the caller-provided expected length and a non-empty appended
token span. It appends exactly that span and continues the existing hash. It is
used only where the generation path establishes monotonic prompt extension;
all other callers retain general `save`.

The offline allocation test now checks full-save fingerprinting, chained
extension fingerprinting against one-shot hashing, rejection of a wrong base
boundary before copies occur, divergent replacement, allocation size, copy
count, and storage retention after clear.

## Test status

Passed:

- `miinfer-reusable-context-allocation-test` built with the host-only CMake
  configuration.
- `ctest --test-dir build/host-only -R reusable-context-allocation
  --output-on-failure`: 1/1 passed.
- All 13 available `host-only` CTest tests passed; the unavailable
  `kquant-wave-host` was excluded because its executable/build target is absent.
- Direct C++20 syntax checks passed for `src/prefill_v2/model.cpp` and the
  updated exact-prefix and snapshot/fork/rollback integration tests.
- Focused `git diff --check` for the reusable-context/model/test files.

Not run:

- Full project test suite and full release gate.
- Model/GPU execution of the newly added exact-prefix cases at 511/512/513/
  1024/1025 tokens and snapshot restore after direct recurrent-state mutation;
  existing cancellation, suffix, and nested fork/rollback integration tests
  are likewise not run.
- MI50 correctness, output parity, thermal validation, or performance tests.

The full HIP CMake configuration was attempted but could not configure: its
pinned ROCm 6.2 linker fails to load `libxml2.so.2`. This is an environment
blocker, not a source compilation result.

The host-only test mocks HIP and cannot certify device state, async ordering, or
real checkpoint contents. The full host-only label also names
`kquant-wave-host`, whose executable/build target is absent; the other 13
registered host-only tests passed. GPU validation remains pending and
prohibited until the live termination-safety path is verified.
