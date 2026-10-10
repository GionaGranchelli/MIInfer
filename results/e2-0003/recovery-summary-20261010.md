# E2-0003 recovery results

Recovery started 2026-10-10 after the first campaign window ended blocked on
SSH to the wrong Z840 address. New endpoints were used: Z840
`fedora-workstation@100.118.66.80`, Machinist `machinist@100.114.213.94`.

## Z840 build and identity

- Host/GPU: `fedora`, AMD Instinct MI50, gfx906, BDF `0000:06:00.0`, unique ID
  `0x21678e17348c2f7`.
- Model SHA-256:
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.
- Runtime: `localhost/miinfer-dev:rocm-7.2.1`, digest
  `sha256:bdc5ed42c985a6a333083e023225f248825a6e1511f38b6b50fbc7d5ace3fe9e`.
- Build flags: Release, HIP gfx906, tests off, benchmarks on. A clean rebuild
  produced the same-source benchmark and operation-reference binary. See
  [clean build log](z840-clean-build.log).
- Same-source benchmark binary SHA-256:
  `55abb51c547314d958b0285005158eb82757edd922f10acf6f018a396bb580e3`.

## Two matched 1K/eight-token pairs

The same binary, model, deterministic prompt, and HIP Graph setting were used
for both routes; only `MIINFER_EXPERIMENTAL_MMQ_GATEUP_ONLY` changed. Both
control/candidate pairs generated identical token IDs. The two independent
pair artifacts, including raw metrics and 1 Hz telemetry, are
[pair 1](z840-pair-1/) and [pair 2](z840-pair-2/).

| Metric | Control median | MMQ-only median | Candidate/control |
|---|---:|---:|---:|
| Prefill | 243.7665 tok/s | 243.6025 tok/s | 99.93% |
| Decode | 28.593 tok/s | 25.521 tok/s | 89.26% |
| Peak sampled VRAM use | 26,032,822,272 B | 18,784,024,576 B | 7,248,797,696 B saved |
| Gate/Up allocation requests | 24,068,487,168 B | 16,938,170,368 B | 7,130,316,800 B saved |

The sampled VRAM number comes from external `rocm-smi` telemetry; allocation
request accounting is reported separately. It does not claim byte-for-byte
whole-model numerical equivalence. Candidate decode also remains above the
20.56 tok/s campaign floor.

All 4 benchmark calls completed with valid metrics, all telemetry samples
present, and clean guard process-group cleanup. Peak junction temperature was
61 C; no warning or stop threshold was reached. The caller sets
`use_hip_graph=true`, the runtime environment sets `MIINFER_HIP_GRAPH=1`, and
the eight-token result exercises graph capture/replay. There is no direct graph
event in the raw metrics; successful graph use is derived from the source path
and completed decode output.

## Operation reference

[gateup-operation-reference.cpp](gateup-operation-reference.cpp) compares the
production-shaped one-token MMQ Gate/Up plus SwiGLU operation against the
existing fused reference kernel using actual layer-0 weights and a deterministic
input. It passed on the Z840 at cosine `0.999998`, relative L2 `0.001991`, and
max absolute error `0.000979885`; pre-recorded limits were cosine `>=0.995` and
relative L2 `<=0.02`. The [guard log](e2-0003-z840-operation-reference.guard.log)
records clean completion. The operation-reference binary SHA-256 is recorded in
[its binary checksum](e2-0003-z840-operation-reference.binary.sha256).

## Separate Machinist qualification

A 4 KiB HIP H2D+D2H copy probe passed on Machinist's MI50 at BDF
`0000:85:00.0`, architecture `gfx906:sramecc-:xnack-`, with junction readings
29–30 C and clean guard cleanup. The earlier reported tiny-copy failure did
not reproduce in this run. This probe is not included in Z840 performance data.
See [probe summary](machinist-copy-probe-20261010.txt), [guard log](e2-0003-machinist-guard.log),
and [image digest](e2-0003-machinist-image-digest.txt).

## Disposition and limits

Candidate A passes the E2-0003 1K/eight-token selection gates and remains
default-off. Candidate B is not triggered because decode retains 89.26% of
control. Whole-model logit equivalence, longer-context behavior, persistent
session and snapshot/rollback behavior, and the full CTest suite were not
checked. Final independent host idle readings are recorded in
[recovery-final-idle-20261010-0813.txt](recovery-final-idle-20261010-0813.txt).
