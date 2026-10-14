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

### HTTPS execution attempt — 2026-10-14

HTTPS fetch succeeded and created a clean detached checkout at
`13418958b5f5146276dc819d59b35418bb911159`. The frozen ROCm 7.2.1 image,
model checksum, GPU identities, and idle state matched preflight. CMake
configuration stopped at `find_package(hipblas CONFIG REQUIRED)` before build
or GPU execution. Evidence is in
`/home/machinist/m31-q32-exec-20261014T092831Z`.

A read-only diagnostic then found the package in that exact image at
`/opt/rocm-7.2.1/lib/cmake/hipblas/hipblas-config.cmake`, with
`libhipblas.so` present. This indicates a CMake search-path issue rather than
an absent dependency. The next configure should pass
`-Dhipblas_DIR=/opt/rocm-7.2.1/lib/cmake/hipblas`. The corrected configure has
not been run; no dispatch trace, full-model correctness result, or timing is
available. Diagnostic evidence is in
`/home/machinist/m31-hipblas-diagnostic-20261014T093053Z`.

The retry with that hipBLAS path advanced one step, then failed because
hipBLAS could not locate its `hip` CMake dependency (`hipConfig.cmake` or
`hip-config.cmake`). No build, trace, or inference started. The next likely
configure input is `-Dhip_DIR=/opt/rocm-7.2.1/lib/cmake/hip`, but that path is
not verified in the image. Raw retry output is in
`/home/machinist/m31-q32-exec-20261014T092831Z/retry-build-run.log`.

### Reference-engine review — 2026-10-14

Reviewed clean local checkouts: mx-llama `71a0624fb36aaaa2df24ccf428236493c813a441`,
gfx906-llama-cpp `1373e889a4d5c69f453fe82c96332585f3d2b960`, and Strata
`a0116b2782a5b82622358c04a06496485d12b2d6`.

- The gfx906 fork gates MFMA on CDNA; MI50 is Vega/gfx906, so an MFMA port is
  not a viable MI50 route. Strata's gfx906 sparse QSA is also a different
  attention algorithm and does not replace dense exact attention.
- The relevant gfx906 reference is the fork's native quantized TILE attention
  (`ggml/src/ggml-cuda/fattn-tile.cuh`, MIT). It avoids a full FP16 KV shadow
  when memory headroom favors native Q8. Its K/V reuse overlaps the current
  MIInfer Q8 dot4 design, so a port is deferred until dot4 is measured.
- The fork's own tuning notes report a bit-identical KQ software pipeline
  regression of 5.8% and geometry/occupancy changes at +0.5%, treated as noise.
  This argues against spending the campaign on manual loop pipelining or a
  speculative geometry sweep.
- No reference source was copied. Strata, the gfx906 fork, and mx-llama are
  MIT. If dot4 fails correctness or speed, the next distinct candidate is the
  fork's compact native-Q8 TILE data path, evaluated on the same 64K prompt.

### Corrected pinned-image build — 2026-10-14

Passing both `hip_DIR=/opt/rocm-7.2.1/lib/cmake/hip` and
`hipblas_DIR=/opt/rocm-7.2.1/lib/cmake/hipblas`, with
`CMAKE_PREFIX_PATH=/opt/rocm-7.2.1`, allowed configure and
`miinfer-m31-fp16-full-model` build to complete in the frozen image. Binary
SHA-256: `57ec6d1cd975e2d7d0ba770eb592f8997e49dfe9acfab2ae02caf68ea4f4031f`.
The trace attempt then stopped before program startup because `libdw.so.1`
was not found. No dispatch trace or inference ran. Logs are under
`/home/machinist/m31-q32-exec-20261014T092831Z`.

To supply that runtime library without changing the ROCm base, Machinist built
an image extending the frozen base digest `sha256:447907c697fb2c39ce9657401e354a70753d5672ad955d4ceb12de976b91224a` with only
Ubuntu Noble `libdw1t64` `0.190-1.1ubuntu0.1`. Derived image digest:
`sha256:422b5d15d384d78cab724a0190ef739d31212674ebc1ae8f0f6cf538b1f5ea8f`.
This derived runtime has not yet passed an executable probe; no GPU run is
claimed.

The corrected rootless Podman probe with SELinux labeling disabled passed:
the mounted binary was visible, `ldd -r` resolved all dependencies including
`libdw.so.1`, and the no-argument invocation returned the expected usage
status. The GPU script then stopped before tracing because host `amd-smi` is
not installed. No trace or inference ran. Use the available `rocm-smi` or
sysfs temperature telemetry for the guard; do not require `amd-smi`.
