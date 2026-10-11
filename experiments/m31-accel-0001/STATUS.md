# M31-ACCEL-0001 live status

Updated: 2026-10-12 00:25 Europe/Amsterdam
Deadline: 2026-10-12 10:00 Europe/Amsterdam

## Source and ownership

- Verified source anchor: `dc83f07f6adaaeec75a344b0155fa049553f2f7b` (`m30/agent-workload-baseline` on Z840); its merge base with `origin/main` is `3991b6bf40f810b63a100c46981e16964bb5434b`.
- Coordinator: branch `experiment/m31-accel-0001-coordinator`, worktree `/tmp/m31-accel-0001-coordinator`, start SHA `dc83f07f6adaaeec75a344b0155fa049553f2f7b`.
- Agent 64K: branch `experiment/m31-accel-0001-agent64k`, worktree `/tmp/m31-accel-0001-agent64k`, same start SHA; implementing only a bounded persistent multi-turn workload. No GPU runs assigned.
- Z840 `/home/fedora-workstation/Development/mi50` remains dirty (tracked M30 workload/result edits plus untracked artifacts/core); do not modify, clean, build, or run from it.
- Machinist `/home/machinist/Development/MIInfer` remains dirty at `d8b833637f85537d6205c9480129d95c819e6d61`; do not modify or build in place.

## Host/resource check (2026-10-12 00:21 local)

- Z840 SSH succeeded. MI50 idle, 29 C junction, 0% use, about 10.9 MB sampled VRAM used. Lake is mounted read/write at `/mnt/lc9-lake`, 1.6 TB free. No active MIInfer/M31 process or tmux session found.
- Z840 campaign source is available only in the dirty M30 checkout. Work must use a separate worktree.
- Machinist SSH succeeded; no active MIInfer/M31 process or tmux session found. Its checkout is dirty; CPU work must use an isolated worktree if needed. Do not use its MI50 for qualification due known H2D issue.

## Current evidence / primary gate

- Highest priority is exactly 65,536 prompt/context tokens followed by at least eight sequential persistent turns with reuse, snapshots where supported, position/state continuity, bounded outputs, and retained per-turn metrics.
- The latest audited native M30 run is 20 turns and passes replay/parity checks at roughly 5.1K cumulative prompt tokens, but capacity is hardcoded to 16K. This is not 64K evidence.
- A separate 64K run is single-prompt/warmed execution and lacks persistent multi-turn reuse; a previous such call reported a 102 C hotspot. Check thermal guards before any new MI50 job.
- Prefix reuse qualification at 8K+512 in M31-0002 diverged at generated token 11 (expected 3299, actual 4434). Therefore persistent suffix performance is unqualified until correctness is established.
- Protocol performance frontier says no active performance PRIMARY after V2-0044 closeout; no performance-sensitive code change is authorized before refreshed attribution. This campaign's user-specified 64K reliability goal is the active campaign priority.

## Queue and next actions

1. Agent returns the 64K workload branch; review the exact diff, build on isolated source, and inspect memory/capacity sizing.
2. Prepare an isolated Z840 worktree at the reviewed candidate SHA; serialize one guarded GPU workload at a time.
3. First attempt: one 64K multi-turn operational/correctness run with retained logs and telemetry. Stop at configured thermal limits. Do not begin an unbounded matrix.
4. Decode/prefill optimization only after a current bottleneck is measured and correctness contract is pinned; all current reported decode/prefill items are historical or parity-limited.
5. LC-0009 stays bounded. Its CPU ordinary-512 result is pushed on `experiment/m31-lc-0009-quantized-correctness` at `a5140f8`; QSUM decode / broader Contract O coverage remain open. Lake inventory hashes are preserved.

## Checkpoint discipline

One GPU job at a time on Z840. Preserve all dirty worktrees and existing logs. Record each run's PID, exact command, source/model/image identity, input IDs, exit status, hashes, temperatures, VRAM and correctness result before starting another.

## 00:30 checkpoint

- Coordinator checkpoint commit: `66e887f78597898f55cc80a7499e0b71b2ebdc20` (pushed to `origin/experiment/m31-accel-0001-coordinator`). This branch publishes the required Z840 committed M30 source ancestry and campaign notes; Z840/Machinist dirty worktrees remain untouched.
- Z840 model SHA-256 was recomputed live and matches `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.
- Local OCI image `localhost/miinfer-dev:rocm-7.2.1` has base label `sha256:fb0de294a2919ff6d503b4845e055c7fb80db5861954694ee5228e48a14411ec`, matching `docs/mi50-environment.md`.
- A persistent base compile of `miinfer-prefix-state-reuse-bench` and `miinfer-v2-0045-qual-bench` is active in `/mnt/lc9-lake/m31-accel-0001-z840`; tmux session `m31_accel_base_build`, container PID 111693, log `/mnt/lc9-lake/m31-accel-0001-captures/build/base-build.log`, exit marker `.../base-build.exit`. It configures/builds only; no MI50 device is passed to the container.
- The existing 64K prefix-state benchmark is unsuitable for this campaign run because one invocation also runs a 128K+512 scenario and exits success even if its printed parity gate fails. The campaign workload must have a dedicated 64K-only path and the result must be manually or programmatically gated on exact parity.
- Decode baseline harness is identified (`miinfer-v2-0045-qual-bench ... decode 2048 128 5`), but it is direct/eager, lacks logits correctness, and current profiling is only a single token on a different graph-disabled route. No decode candidate is authorized from that evidence.

## 00:35 checkpoint — preserved M30 session discovery

The dirty Z840 checkout contains a completed but untracked M30 coding-agent artifact. It has 40 logical turns, passes the minimum-generation gate, fails the visible-answer gate, and reports 66,173 prompt/context tokens by turn 32 and 82,087 by turn 40. It did not start at exactly 65,536, so it does not satisfy the campaign's required setup.

Raw server latency events independently show prefix reuse near turn 40: request 386 had 80,730 prompt / 79,350 reused / 1,380 suffix tokens (1.578 ms restore, 27.717 s suffix prefill); request 390 had 82,087 / 80,730 / 1,357 (1.270 ms restore, 27.933 s suffix prefill). This is better evidence than the script's aggregates, which are misjoined to latency events. The per-turn client completion/exit metadata is absent, and one frozen output gate failed; treat it as diagnostic only.

The original M30 artifacts were copied without modification to `/mnt/lc9-lake/m31-accel-0001-captures/z840-m30-handoff/`; Lake-side SHA-256 inventory is retained there. Z840's dirty source, JSON outputs, logs, graph reflections and 13 GB core dump remain untouched. The worker has been told to avoid that checkout and to collect metrics inline in the campaign harness.

Build note: preset-only CMake configure failed to find `hipblasConfig.cmake`; the pinned image does contain `hipblas-dev` and the config at `/opt/rocm-7.2.1/lib/cmake/hipblas`. The documented qualified-host command sets `CMAKE_PREFIX_PATH=/opt/rocm-7.2.1`. A clean, separate build with this path is now active in the coordinator worktree; no GPU device is mounted.

## 00:40 checkpoint — baseline build and recovered session

- Coordinator evidence update `269d8ef` is pushed. A second worker branch `experiment/m31-accel-0001-suffix64` is isolated at `dc83f07`; it is adding an explicitly selected 64K-only prefix-reuse benchmark mode with fail-closed correctness checks. No GPU run assigned to that worker.
- The qualified-container compile now passes after applying the documented ROCm prefix: `cmake -S . -B build/m31-accel-0001 -DCMAKE_PREFIX_PATH=/opt/rocm-7.2.1 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_HIP_ARCHITECTURES=gfx906 -DMIINFER_ENABLE_HIP=ON -DMIINFER_HIP_ARCHITECTURE=gfx906 -DMIINFER_TARGET_ARCH=gfx906 -DMIINFER_BUILD_TESTS=ON -DMIINFER_BUILD_BENCHMARKS=ON && cmake --build build/m31-accel-0001 --parallel 2 --target miinfer-prefix-state-reuse-bench miinfer-v2-0045-qual-bench`. Exit marker: 0.
- Source for that build was the clean Z840 worktree at `66e887f` (runtime tree exactly `dc83f07` plus campaign docs). Binary SHA-256: prefix bench `e74e4834c0d629bd7efb074e9f81c51f4ca670d63430eb83fad2f008b2e3afe2`; decode bench `a0b9b0bdf7f9e0fa1ddaf5e6d9ec93000e69591cc1218a466463dfd00c7b0019`.
- Post-build hardware check: no tmux session or MIInfer process; GPU remains idle at 28 C junction, 0% busy, ~10.9 MB VRAM.
- Dirty-source M30 JSON/log copy hashes match their originals; see `z840-m30-handoff` on Lake. The untracked results are not committed to Git because their completion gate failed and they contain the full generated transcript; only the diagnostic summary is tracked here.

## 00:45 checkpoint — guarded 64K diagnostic attempt

- Worker commit `9ab12bc` adds an in-process 1 GiB free-VRAM floor and per-turn `total_ms`, decode throughput, and sampled VRAM to the synthetic-token native harness. Coordinator cherry-pick `e1c3ee7` is pushed; the updated HIP target built successfully on the clean Z840 worktree with exit 0.
- Guarded attempt source: `e1c3ee7e9b6b7fc92ae2490d08d74c60abee67a5`; model SHA: `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`; image: `localhost/miinfer-dev:rocm-7.2.1`; selected device: `ROCR_VISIBLE_DEVICES=0`.
- This was the synthetic-token persistence diagnostic, not a coding-agent workload. It loaded the model and started the exact-64K prompt prefill. It had not emitted a completed turn when junction temperature crossed the 85 C warning point. Manual stop was issued; sampled peak junction reached 87 C, and sampled peak VRAM use was 30,387,605,504 B (34,342,961,152 B total). Idle afterward: 34 C junction, 10,854,400 B VRAM used, no MIInfer process.
- Attempt capture is `/mnt/lc9-lake/m31-accel-0001-captures/native64k/`. Telemetry SHA-256 is `741006b9528a19d97b5c021d3790c765c83ebf713a3c29144bdc54910b55d593`; environment SHA-256 is `5bc3ddf5a6298b6bfb56883e88d71dab35723e4c5a1bbe0cd7e2f94623ca3e01`; guard note SHA-256 is `0c6895dcf73c940c6b440391d6e89fb37350c23571aff893803702946305d835`.
- The temporary remote wrapper had an exit/logging race: its marker says 0, while retained Podman inspection showed container exit 137 after the manual stop; stdout was empty because the native harness had not flushed a turn. A tracked replacement, `experiments/m31-accel-0001/run-guarded.sh`, now retains/follows container output, waits for final state before reading the exit code, checks the pinned model SHA, logs a warning at 85 C, hard-stops at 90 C or the VRAM floor, and passes boundary self-checks. Native turn output now flushes immediately.
- Production-path workload checkpoint: `82f6cd5` adds a CPU-only MIInfer tokenizer counter and an eight-turn M30-style tool workload. On Z840, CPU-only CMake build passed and `--self-check` counted the constructed initial ChatML request at exactly 65,536 tokens from seven repository files; the GPU remained idle at 28 C. Self-check log SHA-256: `a7ad2173803692c76242352c56171899357f323901f7bcccd8010543f4e260a4`.
- A first attempt to run that counter in Podman could not access the model bind mount and failed before tokenization; running the same self-check directly on the host passed. No production-server/GPU workload has run yet.
- The `miinfer` serving target now also builds on the clean Z840 worktree (exit 0; binary SHA-256 `106feada3db19dca0d8b02bd9f5612e647b9175efabf3553ebc99a759d77e9a8`). The tracked runner can start the production server, wait for `/readyz`, then run the exact-token workload in the same monitored container. Its 131,072 engine capacity is headroom for the 64K start and turn history only; the workload is not a 128K prompt or performance run.
- Next: review/push the runner mode and command documentation. Before any new GPU request, inspect thermal behavior and keep an operator on telemetry; the synthetic attempt reached 87 C quickly, so stop if temperature continues rising toward 90 C.
