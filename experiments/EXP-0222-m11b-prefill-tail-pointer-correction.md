# EXP-0222 — M11-B Final Chunk Pointer Correction

## Question

Does the layer-major prefill path return the final token's hidden-state slot
when the prompt length is not a multiple of the fixed B=4 chunk size?

## Finding

No. `prefill_layer_major()` returned slot 3 of the final buffer for every
prompt of at least four tokens, even when the final chunk contained fewer than
four tokens. P513 therefore passed stale slot-3 data to the LM head and the
first decode step.

The return pointer now uses `count - 1` for the final chunk. The change does
not alter layer execution or prefill workspace allocation.

## Verification

- Build: `mi50-release` succeeded.
- CTest: all 21 tests passed.
- Corrected P513 production-shaped prefill-only runs: `11177.20 / 11177.11 /
  11231.21 ms`, median `11177.20 ms`, `45.90 tok/s`.
- A corrected P513 run with one generated token completed successfully at
  `45.51 tok/s` prefill.

## Decision

**KEEP.** This is a correctness fix, not a performance optimization. The
M11-B 100 tok/s gate remains open.
