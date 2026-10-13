# EXP-0404 — M31 FP16 tiled full-model prefill

**Status:** IN PROGRESS

**Baseline:** `88fd63eb0cf8b765f37e5f52168c80783df62e7b`

## Objective

Determine whether the existing query-tiled attention improvement transfers to
full-model prefill with FP16 KV storage, and account for the 64K cache's VRAM
cost. Run matched deterministic 8,192-token prompts with Q8/Q8 control and
FP16/FP16 candidate, capacity 65,536. The candidate enables the existing
opt-in `MIINFER_V2_0043_GQA_ATTENTION` path.

## Gates

- Verify one MI50 BDF and record each mode's static VRAM allocation.
- Record cold prefill wall time and final-logit cosine/max error/top-1 match.
- Keep this as product progress only if FP16 fits with safe VRAM headroom and
  materially lowers full-model latency. Then run the established real coding
  task and the 64K workload before any production default change.

## Result

Pending.
