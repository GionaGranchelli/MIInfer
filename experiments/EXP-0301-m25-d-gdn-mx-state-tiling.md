# EXP-0301 — M25-D mx-style GDN state tiling

**Status:** REJECT
**Milestone:** M25-D
**Date:** 2026-09-11

## Hypothesis

The pinned mx-llama GDN mapping—Wave64 lanes covering contiguous state rows,
two state columns per warp, and one block per value head—could reduce the
recurrent P512 causal-core cost on gfx906.

## Baseline and candidate

The baseline was the qualified M23 resident-all schedule using eight ordered
64-token GDN chunks. The candidate was an opt-in kernel that copied the mx
state-row tile geometry while retaining MIInfer's token-major tensors and
in-place transposed state storage.

The candidate was never made the default and was removed after validation.

## Environment

```text
GPU: AMD Instinct MI50 / gfx906
SCLK/MCLK: 1606/1000 MHz (continuous telemetry)
Model: Qwen3.8-27B-Q4_K_M.gguf
Prompt: exact 512-token repeated fox sentence
Resident allocation: 22,801,772,884 B (baseline), 22,919,836,628 B (validation)
```

## Results

The exploratory non-validation candidate run completed in 4,867.91 ms
(105.18 tok/s), versus 5,050.18 ms for the single baseline run. This was not
treated as a qualified performance win because it was one A/B pair and the
candidate failed the state contract.

The validation run reported:

```text
output_max_abs=0.000955582
qkv_max_abs=3.8147e-05
gate_max_abs=6.67572e-06
state_max_abs=26.1423
history_max_abs=2.28882e-05
```

The canonical resident path's state error is approximately `2.28882e-05`.
The large candidate state error makes the apparent output agreement
insufficient; recurrent state is persistent and must match before timing is
meaningful.

## Interpretation

The mapping computes reductions over the wrong state dimension for MIInfer's
Qwen3.8 recurrence. Its shape resembles mx-llama's kernel, but the external
state/value/output contract is not interchangeable. This is the same failure
class as the previously rejected row-wave and column-tiled experiments
(`EXP-0154`, `EXP-0166`).

## Decision

**REJECT.** The candidate code was removed. Keep the qualified 64-token GDN
path and do not retry this transposed state-row mapping without a standalone
state-contract proof first.

## Follow-up

Continue M25 differential work at the kernel-contract level. The next GDN
experiment must validate one recurrent state update independently before any
full-model timing is collected.
