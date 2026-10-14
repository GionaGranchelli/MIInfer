# EXP-M31-FP16-GQA-Q32-CTA-SCREEN

## Hypothesis

Use a 32-token × 2-query-head CTA for FP16 GQA prefill. The existing 16-token × 4-head candidate carries 64 row slots but masks 25% of them because each KV head owns six query heads. The 32 × 2 layout fills all rows. For each 32-token span and KV head, it maps to three CTAs instead of four in the 16 × 4 candidate (25% fewer CTAs and repeated K/V tile staging). This is a scheduling estimate, not a measured speedup.

## Change

Added opt-in `MIINFER_M31_FP16_GQA_Q32_CTA=1` to the existing FP16 fragment-reuse launcher. The path uses 512 threads, 64 query rows, 32 query tokens, and three two-head groups per KV head. It retains the existing K/V tile, FP16 arithmetic, split count, and per-row causal/split masks. It cannot be enabled at the same time as `MIINFER_M31_FP16_GQA6_WIDE_CTA`.

## Compile and code-object screen

- Target: gfx906, wave64, 512-thread workgroup.
- Build: `miinfer-m31-fp16-full-model` succeeded with the local ROCm 7.14.60850 image (`sha256:622c025824d465faa0f5e59eea4f723845032011df273c3b11e25031965fc428`). This is a compile screen, not the campaign's pinned ROCm 7.2.1 environment.
- Q32 candidate metadata: 71 VGPR, zero VGPR spills, zero SGPR spills.
- Existing 16 × 2 specialization metadata from the same compiler: 66 VGPR, zero spills. Existing 16 × 4 candidate: 63 VGPR, zero spills.
- Dynamic LDS requested by the launcher: 45,824 bytes per CTA. The code object compiled for gfx906.

## Correctness and performance status

The row/token/head mapping and split coverage passed an assert-based host-side check. A source review found no uncovered or duplicated valid query rows. No MI50 execution, full-model logits comparison, or timing has been performed; local ROCm reports no initialized AMDGPU driver. Keep this candidate opt-in until a pinned-host 2K/full-logit gate and matched prefill measurement pass.
