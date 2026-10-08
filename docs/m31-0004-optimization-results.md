# M31-0004 — Optimization results

## Status

**CODE_COMPLETE / GPU_VALIDATION_PENDING.** No hardware correctness,
performance, thermal, or output-parity claim is made. GPU tests remain deferred
until the live termination-safety gate is verified.

Starting code base: `849b14b774a0a4cae9d8b7f169b670412d6132ae` (M31-0003).
Runtime implementation commit: `b9533bc66d6f7aa3193aac4baa745f46d1255cc7`.

## Implemented changes

1. Invalidate resident checkpoint/snapshot metadata before model-state mutation
   so a partially completed prefill/decode cannot leave a stale resident flag;
   save the full prompt on successful completion, including short/partial
   prompts, while avoiding duplicate capture at an already-saved final boundary.
2. Add explicit incremental checkpoint token extension for proven monotonic
   prompt growth. This removes repeated full-prefix host token copying and
   hashing in the cached generation path.
3. Preserve all 512-boundary device checkpoint captures, snapshot ownership,
   stream ordering, and generic full-prefix save behavior.

## Work reduction ledger

| Operation at 128K | Before | After | Reduction |
| --- | ---: | ---: | ---: |
| Host hash visits across 256 boundaries | 16,842,752 | 131,072 | 16,711,680 (about 99.2%) |
| Explicit token payload copied into cached vector | 16,842,752 | 131,072 | 16,711,680 (about 99.2%, excluding occasional capacity relocation) |
| GDN D2D checkpoint calls | 24,576 | 24,576 | 0 |
| Aggregate GDN checkpoint bytes | 40,667,971,584 | 40,667,971,584 | 0 |
| GDN kernel arithmetic / launches | unchanged | unchanged | 0 |

Copy-volume figures are source-derived totals. They are not measured bandwidth
or elapsed-time savings. No device copies were proven unnecessary: boundary
captures preserve the latest completed prefix across interruption; the single
slot does not retain every earlier intermediate prefix.

## Remaining work and validation

The strongest remaining computational opportunity is improving Q/K data reuse
across GDN state-column tiles. It requires a measured gfx906 experiment; TC4
currently has a 1.63% slower P512 median and is not selected. Full GDN parallel
scan, kernel fusion, and sync removal remain speculative.

Validation: 13/13 available host-only tests passed when excluding the registered
but unavailable `kquant-wave-host`; the focused allocation test is among those
13. The modified model and updated integration-test translation units pass
direct C++ syntax checks. The full HIP CMake configure is blocked because the
pinned ROCm 6.2 linker cannot load `libxml2.so.2`. Required later GPU checks are: state/output parity at
boundary lengths 511/512/513/1024/1025, cancellation before and after a boundary,
prefix extension, nested snapshot/fork/rollback, then short guarded 8K/32K
integration and a direct checkpoint-operation microbenchmark. 64K/128K should
only follow if the short tests and safety gate pass.
