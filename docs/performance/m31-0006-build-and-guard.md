# M31-0006 — Build and stop-guard status

Status: build, stop control, targeted GPU regressions, bounded attention
baseline, and short production graph smoke passed. No production optimization
was promoted because every tested reduced-split candidate lost.

## Reproducible build

- Starting HEAD: `54d2fbed6e1b5f8dd8ff7c7293bff5144fe09fc7`; the shared worktree
  contains other user changes and was not cleaned or reset.
- P620 native ROCm 6.2.41133 / clang 18 configure remains blocked before project
  compilation because LLD needs `libxml2.so.2` and the host has `.so.16`. No
  SONAME shim or ABI substitution was made.
- Build host: Z840; pinned `localhost/miinfer-dev:rocm-7.2.1`, manifest digest
  `sha256:bdc5ed42c985a6a333083e023225f248825a6e1511f38b6b50fbc7d5ace3fe9e`;
  HIP runtime/driver `70253211`, clang/LLD 22. The CMake HIP presets set
  `CMAKE_PREFIX_PATH=/opt/rocm-7.2.1`.
- The full HIP Release build and 14/14 host-only tests passed on the dirty
  current-source snapshot. Original logs are under
  [current-tree-build](../../results/m31-0006-current-tree-build/); host-test
  log SHA-256 `cfc23431e5ff2b287e4ffb9e26c2c4c0c4503d1cbaaab550f9cde3226e8094d9`.
- Final build-input archive:
  [source-snapshot.tar](../../results/m31-0006-current-tree-build/source-snapshot.tar),
  SHA-256 `47a49541c13a06f8c3d2e2aa5fa566ebc385a5d7459fc350ed8649e7445ef37d`.
- After the final harness edits, the M31-0004 exact-prefix, snapshot/fork and
  agent-branch tests, M31-0005 decode-boundary test, GPU guard worker,
  graph-attention microbench, and production graph smoke all rebuilt
  successfully from that archived source. Final targeted build log SHA-256
  `3afb924cb0259c59810115988ec1382467a200e6146354eaa311aa98df51e09d`.
- Model: `Qwen3.8-27B-Q4_K_M.gguf`, 17,106,775,008 bytes, SHA-256
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.

## Stop guard

The CPU fault matrix passed: TERM cleanup, TERM-ignoring nested tree and KILL
escalation, monitor/poll faults, missing/stale telemetry, natural exit,
descendant cleanup, and unrelated-sentinel isolation. Cooperative cleanup was
80–90 ms; TERM-ignoring tree cleanup completed in about 2.14 s; monitor poll
failure cleanup took about 110 ms. Raw log SHA-256
`e07be748568c370cf321dce5dbb4b7a226bd90a7fbd0609e7093bb20ebd3e1c8`.

The live proof used the Z840 MI50 at BDF `0000:06:00.0`, exact gfx906. A worker
allocated/touched 4 MiB; an injected 85°C sample triggered TERM 10 ms after
detection. The worker exited 140 ms after detection and its process group was
confirmed empty at 170 ms. KFD/VRAM returned to zero. Actual junction stayed
near 28–29°C. Raw log SHA-256
`39ebcb7c6d6166b6be344375a2e5d1e25b37733c46bfc869bde968ce87665b20`.

Physical cooling has been confirmed by the user; no cooling investigation or
real-threshold heating test was reopened. Guard limits remained 80°C warning /
85°C stop.

## GPU evidence and remaining boundary

The M31-0004 exact-prefix, snapshot/fork/rollback, agent-branch regressions and
M31-0005 eager/graph KV-boundary regression all passed; see
[pending-regressions](m31-0006-pending-regressions.md). Direct graph/eager
parity passed at positions 1, 511 and 512. The synthetic 8K/32K/64K/128K
attention cases passed CPU-oracle and future-slot checks. The short real-model
graph smoke passed with four generated tokens. All runs were sequential and
left no residual KFD process or VRAM allocation.

No full 128K model prefill was run. The 128K result is direct attention shape
only; no long-context end-to-end decode or production speedup is claimed. The
remaining engineering question is memory traffic attribution, not hardware
availability or thermal safety.
