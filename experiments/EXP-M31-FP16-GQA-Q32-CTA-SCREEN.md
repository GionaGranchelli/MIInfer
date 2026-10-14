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

## Machinist execution preflight — 2026-10-14

The live Machinist preflight identified two idle `gfx906` MI50s: GPU0 at
`0000:04:00.0` (UUID `0x21678e17348c2f7`) and GPU2 at `0000:88:00.0` (UUID
`0x2152414173497dd3`). Both were at 180 W power caps and 22–24 °C, with no
KFD compute PIDs. The device at `0000:85:00.0` is not an MI50. The model
SHA-256 matched `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.

The available container reports ROCm 7.2.1 / HIP 7.2.53211, image ID
`50ff28a28cc13a3ca7333cd0c70e322d16245967c5fdf0d30a1bf57bda80a512`, and
repo digest
`sha256:447907c697fb2c39ce9657401e354a70753d5672ad955d4ceb12de976b91224a`.
This exact image identity can be frozen for a matched pair.

No build, trace, or inference run started. Fetching the pushed source branch
from the Machinist checkout failed because its `origin` uses GitHub SSH and
the host returned `Permission denied (publickey)`. The existing remote
worktrees are dirty and are not at this candidate SHA. The preflight evidence
is stored on Machinist at
`/home/machinist/m31-pq32-final-20261014T092317Z`.

The Q32 candidate therefore remains compile-screened only. A matched
65,536-token control/candidate run, full-logit comparison, and dispatch trace
remain outstanding; the failure was source access, not candidate correctness
or performance.
