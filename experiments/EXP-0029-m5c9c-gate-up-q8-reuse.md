# EXP-0029 — M5-C9c Gate/Up activation-Q8 reuse

**Status:** CLOSED — KEEP; PRODUCTION-SELECTED
**Milestone:** M5
**Date:** 2026-09-01
**Baseline commit:** `c1a35db`
**Candidate:** production default `MIINFER_FFN_Q8_REUSE=shared`; control
`MIINFER_FFN_Q8_REUSE=separate`
**Result commit:** `1159a89`

## 1. Question

Do Gate and Up independently quantize the same FFN-normalized activation, and
can one unchanged Q8_1 buffer be quantized once and reused by both GEMVs?

## 2. Candidate

The existing separate path quantizes the same `ffn_norm` source for Gate and
Up in two sequential projection calls. The opt-in candidate quantizes once,
then invokes the unchanged Gate and Up projection kernels with that existing
Q8 buffer as prequantized input. The candidate supports both the standard and
exact metadata contracts, but rejects mismatched Gate/Up input contracts.

No GEMV arithmetic, quantization implementation, precision boundary, weight
format, kernel geometry, or Gate/Up concurrency was changed.

## 3. Verification design

With `MIINFER_VERIFY_FFN_Q8_REUSE=1`, the candidate performs a diagnostic
second quantization into a separate persistent buffer using the old Up call
contract. It copies both complete block streams to the host and compares all
payload and metadata bytes. Any mismatch aborts the run. The position audit
reports `gate_up_q8_reuse_checks` and `gate_up_q8_reuse_mismatches` per audited
profiled position. Verification is intentionally excluded from performance
measurements because it adds synchronous host-visible copies.

## 4. Acceptance gates

The pinned real-model gates passed:

* Gate/Up Q8 streams byte-identical across representative positions;
* Release CTest 19/19;
* identical 64-token deterministic trajectory;
* reset/replay determinism;
* repeatable P64 interleaved A/B performance result;
* no allocation, cache, or precision-contract regression.

The verifier tested positions 1, 8, 16, 32, and 64. Each position performed
36 complete Gate/Up block-stream comparisons, for 180 checks total, with zero
mismatches. The position-audit output is retained at
`bench/results/20260901T133416Z-397585/verification-position-audit.json`.

The production default was switched to shared reuse only after these gates
passed; the separate path remains available as an explicit control.

## 5. Benchmark commands

```bash
cmake --preset mi50-release
cmake --build --preset mi50-release \
  --target miinfer-qwen3-attention-ab-bench miinfer-qwen3-position-audit

scripts/run-m5c9c-gate-up-q8-ab.sh \
  /path/to/Qwen3-8B-q4_0-b968826d.gguf

MIINFER_FFN_Q8_REUSE=shared \
MIINFER_VERIFY_FFN_Q8_REUSE=1 \
build/mi50-release/miinfer-qwen3-position-audit \
  /path/to/Qwen3-8B-q4_0-b968826d.gguf \
  --positions 1,8,16,32,64 --gpu-argmax \
  --json-output /tmp/m5c9c-position-audit.json
```

## 6. Performance results

The balanced three-pair real-model A/B used 64 measured growing-context
decode tokens. Both paths produced the same 64-token trajectory:

| Path | Mean decode | Throughput |
|---|---:|---:|
| Separate | 1173.233789 ms | 54.550083 tok/s |
| Shared | 1159.999257 ms | 55.172449 tok/s |

The shared path improved throughput by **1.14%**. Structural accounting shows
the Gate/Up input quantization is reduced from two launches per layer to one;
the shared run reports 1553 dispatches/token. Hardware telemetry observed
approximately 930 MHz SCLK and 350 MHz MCLK, so the result is a same-state A/B
comparison rather than a canonical-clock absolute benchmark.

## 7. Decision

```text
KEEP — production-select shared Gate/Up Q8 activation reuse
```

The default fast decode policy now uses the shared buffer. The separate path
remains available as `MIINFER_FFN_Q8_REUSE=separate` for regression A/B tests.

## 8. Follow-up

Refresh the production P64 attribution with shared reuse enabled and choose the
next optimization from the updated FFN/token profile. Retain the long
trajectory gate for future arithmetic or materialization changes.
