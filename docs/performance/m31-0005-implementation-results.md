# M31-0005 — Implementation results

Status: PARTIAL — source attribution and narrow eager-path correctness fix
complete; GPU validation and HIP compilation pending.

## Investigated production paths

- M31 benchmark uses graph decode by default. The graph captures embedding,
  16 topology blocks, final norm, and logits. The GQA path uses a dynamic
  Wave64 Split-K attention kernel with a fixed 64-split launch grid.
- Graph stage 1/2 are two same-stream launches per attention layer, with no
  host sync between them. 48 recurrent and 16 GQA blocks execute in the graph.
- Non-graph/eager decode instead uses suffix Split-K and the prefill scratch
  workspace. Its 3-vs-32 split allocation is not the graph frontier setting.
- Decode host control copies logits D2H, synchronizes for CPU sampling, then
  uploads the next state. No per-layer synchronization was found in normal
  graph decode.

## Work and memory model

For 16 GQA layers, 24 Q heads, 4 KV heads, and D=256, exact QK+PV work is
`196,608 * context_length` MACs/token: 1.61B at 8K and 25.77B at 128K. Unique
FP16 K+V payload is 64 KiB/token across layers: 0.5/2/4/8 GiB at
8K/32K/64K/128K. Six query heads share each KV head; 6x is a logical read
request ceiling, not measured HBM traffic.

Graph Split-K scratch is about 1.51 MiB accumulators plus 12 KiB metadata.
Eager workspace reserves 387 MiB at 32 splits or 36.3 MiB at 3 splits, a
350.6 MiB difference. That eager reduction was already present and does not
explain graph decode scaling.

Measured aggregate decode latencies are 38.2/47.4/62.1/90.1 ms/token at
8K/32K/64K/128K. The ~51.9 ms/token 8K→128K difference is not decomposed into
components; no measured component breakdown is available.

## Implemented

1. Corrected the eager one-token suffix caller's base position so the kernel's
   inclusive `base_position + token + 1` does not include a future KV slot.
2. Added an adversarial GPU regression target/test with a poisoned future KV
   position. It is registered as GPU-required, not host-only.
3. Kept the `kquant-wave-host` CTest registration inside the HIP-enabled branch,
   matching its executable's availability and preventing host-only CTest from
   advertising a missing target.

No graph attention/split, KV precision/layout, snapshot, or sampling behavior
was changed. No speedup is claimed. Before/after performance is not available.

## Validation and blockers

- Host-only configuration succeeds; all 13 host-only tests passed.
- C++ syntax-only checks passed for the edited attention source and new test.
- HIP configuration was attempted but the host ROCm 6.2.41133 linker fails to
  load `libxml2.so.2`; only `libxml2.so.16` is installed. No unsafe soname
  symlink or toolchain substitution was made.
- The pinned container image is not cached; available root space was about
  8.1 GiB, so a potentially large pull was not attempted.
- Full host-only application build fails because the existing CLI includes
  `hip/hip_fp16.h` without HIP headers in that configuration. The targeted host
  test executables build and pass.
- GPU test and inference were not run. The live stop guard has not proved
  timely termination during real inference.
- Changes have not yet been committed; the worktree contains extensive
  pre-existing user-owned M31 edits and artifacts. Stage/commit only exact
  M31-0005 paths and hunks.

## Candidate decision and next bottleneck

The confirmed eager bound error is fixed, but it is outside the graph path that
produced the M31 latency frontier. The most credible graph-path experiment is
an exact-parity active-split comparison (default versus one lower fixed split
count) at 8K then 32K, not another full profile. GQA head reuse is a higher-risk
follow-up because actual HBM duplication is unknown. Q8 is not promoted as a
speed optimization after the historical 64K regression.

Next bottleneck: establish source-level device timing for graph GQA attention
versus recurrent work, and validate the real workload stop path before any
GPU execution. The physical fan setup is user-confirmed as okay; that does
not cure the outstanding safe-stop evidence gap.
