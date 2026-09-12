# EXP-0318 — M25 attention bakeoff pipeline isolation

**Status:** KEEP  
**Milestone:** M25  
**Date:** 2026-09-12  
**Baseline commit:** `e0bae93`  
**Candidate:** working tree only

## Finding

`bench/m24_attention_layer_bakeoff.cpp::set_environment()` cleared
`MIINFER_MX_PIPELINE` with the other experimental selectors, then re-enabled
it unconditionally. That forced the previously rejected M25-C register
pipeline into the control and H/I attention bakeoff modes.

## Fix

Remove the unconditional `setenv("MIINFER_MX_PIPELINE", "1", 1)`. The
benchmark now leaves the selector unset, so the qualified staged MMQ kernel is
used. The function still clears the selector first, preventing ambient shell
state from selecting it.

## Verification

The Release bakeoff target rebuilt successfully. The full Release CTest suite
passed `24/24` after the fix. The rejected pipeline remains available only for
explicit isolated tests through its environment selector.

## Decision

**KEEP.** This is a benchmark correctness/reproducibility fix, not a throughput
claim. Any new pipeline comparison must opt in explicitly and identify itself
as the M25-C candidate.
