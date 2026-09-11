# EXP-0288 — M23 direct P512 GDN scan

**Status:** REJECT
**Milestone:** M23-R
**Date:** 2026-09-10

## Hypothesis

Replacing eight ordered 64-token GDN chunk launches with one register-resident
P512 scan will reduce causal scheduling overhead while preserving state.

## Candidate and correctness

`MIINFER_PREFILL_GDN_DIRECT=1` selects `launch_m12_gdn_direct` for a complete
wide projection batch. With `MIINFER_WIDE_VALIDATE=1`, the exact P512 model
run matched the canonical recurrent layer closely:

```text
qkv_max_abs=3.8147e-05
gate_max_abs=6.67572e-06
state_max_abs=2.28882e-05
history_max_abs=2.28882e-05
output_max_abs=0.000955582
```

## Result

On the MI50/gfx906 Qwen3.8-27B Q4_K_M P512 schedule, the non-validation run
completed in 28,565.28 ms (17.92 tok/s), versus the causal 64-token path at
8,734.09 ms (58.62 tok/s). The direct kernel is therefore 69.4% slower.

## Decision

REJECT for production. Keep the implementation as an opt-in correctness probe;
retain 64-token causal chunks while projection width remains independently
wide. Do not tune this kernel before the P512 matrix profile identifies a
causal-scan bottleneck.
