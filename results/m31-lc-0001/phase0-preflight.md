# M31-LC-0001 Phase 0 — Preflight

- Qualification branch: `experiment/m31-lc-0001-context-scaling`, based on pushed commit `8a8591a18f0ab66b6edae3ea42074a738a349969`.
- Worktree is source-bound and was clean before the harness-only commit `667f3e7db7329529b95c8cec785c24bc1d4c9226`.
- SSH to `fedora-workstation@100.118.66.80` succeeded within the two permitted preflight attempts. The first command's 30-second bound expired during model hashing; the second completed the hash.
- Host: `fedora`; GPU index 0 is AMD Instinct MI50 (`gfx906`), BDF `0000:06:00.0`, unique ID `0x21678e17348c2f7`. Live idle sample: 28 C junction, 0% GPU busy, no KFD PIDs.
- Pinned image digest matches `sha256:bdc5ed42c985a6a333083e023225f248825a6e1511f38b6b50fbc7d5ace3fe9e`.
- Model is 17,106,775,008 bytes with SHA-256 `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.
- Existing E2-0003Q evidence at the source baseline records a clean Release build and full CTest result of 34 passed, 0 failed, 1 skipped (`package-archive-smoke`). The six focused smoke tests were also rerun on Z840 under the 80 C warning / 85 C stop guard: route parser/default-off, HIP smoke, quantized KV attention, M31 decode-attention bound, cached-attention determinism, and reusable-context allocation; all six passed.
- The thermal/process guard self-test passed locally, including injected 85 C abort, bounded worker cleanup, and stale telemetry handling.
- The runtime benchmark explicitly configures greedy sampling with temperature 0, top-p 1, top-k 1, repetition penalty 1, frequency/presence penalties 0, stop IDs disabled, repeat-last-n 256, seed 42, and HIP graph enabled. Default-off behavior is covered by the parser smoke and E2 evidence.

Raw host output is in `phase0-smoke-final.ctest.log` and `phase0-smoke-final.guard.log`. The prior 1K E2-0003Q qualification is reused; no 1K A/B run was repeated.
