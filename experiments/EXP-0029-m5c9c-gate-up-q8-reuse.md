# EXP-0029 — M5-C9c Gate/Up activation-Q8 reuse

**Status:** IMPLEMENTED — OPT-IN CANDIDATE; REAL-MODEL VALIDATION PENDING
**Milestone:** M5
**Date:** 2026-09-01
**Baseline commit:** `c1a35db`
**Candidate:** `MIINFER_FFN_Q8_REUSE=shared`

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

Pending execution on the pinned real model:

* Gate/Up Q8 streams byte-identical across representative positions;
* Release CTest 19/19;
* identical 64-token deterministic trajectory;
* reset/replay determinism;
* repeatable P64 interleaved A/B performance result;
* no allocation, cache, or precision-contract regression.

The existing production default remains the separate path until all gates
pass.

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

## 6. Decision

```text
PENDING — candidate implemented; hardware verification and A/B results not yet run
```

## 7. Follow-up

If the verifier and 64-token trajectory pass, compare the fresh P64 profile
against C9a/C9b and select the candidate only for a repeatable end-to-end
gain. If either byte identity or trajectory fails, retain the separate path
and investigate the specific contract or lifetime mismatch.
