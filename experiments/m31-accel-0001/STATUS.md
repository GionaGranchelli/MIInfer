# M31-ACCEL-0001 live status

Updated: 2026-10-12 03:10 Europe/Amsterdam
Deadline: 2026-10-12 10:00 Europe/Amsterdam

## 2026-10-12 03:04 Z840 SSH and capture recheck

- SSH succeeded on the first attempt. Lake is mounted read/write with 1.6 TB free. The `agent64k-04` capture contains no saved request JSON or completed runner result; server request 3 was queued and the recorded container exit is 137.
- MI50 is idle at 27 C junction, 0% GPU use, 20 W, and 10.9 MB VRAM. Fan telemetry remains 15% / 0 RPM. No inference ran; airflow through the passive heatsink remains unconfirmed, so the GPU gate stays closed.
- Next: establish safe airflow/cooling, run guarded production 64K correctness, then collect the prepared cold/suffix performance comparison. Decode/prefill results, 64K state continuity, and LC-0009 QSUM decode evidence remain unqualified.

## 2026-10-12 03:07 cursor validation boundary

- Source/call-site audit found no direct `decode_step()` test. The recurrent state view copies its cursor, and HIP recurrent decode uses that cursor for convolution-history addressing. Host-only checks cannot validate the fix; it requires a guarded GPU state check on Z840 after cooling is confirmed.
- Recursive scan found no recoverable files in the old `agent64k-04/session-cache` or request/prompt artifact paths. The new runner records request bodies for the next attempt.

## 2026-10-12 03:08 persistent-session path review

- Production launcher enables `MIINFER_SESSION_REUSE=1` and a persistent `--session-dir`; the streaming handler captures processed prompts and restores matching prefixes. Per-request logs expose restore time, GQA reused tokens, and GDN checkpoint/final positions. This is ready for the guarded agent run but not live-qualified.
- Z840 remained idle at 27 C/0% use with fan telemetry 15%/0 RPM; Lake has 1.6 TB free. No GPU work ran.

## 2026-10-12 03:10 cooling configuration check

- Live sysfs confirms the host is an HP Z840. HP's manual says the 1125 W PSU configuration has two front system fans and supports up to two 300 W compute GPUs; AMD lists the MI50 as passive-cooled at 300 W peak.
- These specifications do not verify this unit's PSU/fan configuration or current airflow through the card. The MI50 remains idle at 27 C with fan telemetry 15% / 0 RPM. Waiting for physical airflow confirmation; no GPU run started.

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

## 01:22 checkpoint — Z840 access restored

- SSH to `fedora-workstation@100.118.66.80` succeeded. Lake is mounted read/write at `/mnt/lc9-lake` with 1.6 TB available; `/` has 11 GB free. The MI50 is idle at 28 C junction, 0% busy, 10,854,400 B VRAM used, and no MIInfer process was found.
- Cooling remains unresolved: `rocm-smi` still reports fan level 15% and 0 RPM. This is not physical confirmation of airflow, so no sustained GPU workload was started.
- Coordinator workload now writes a partial result before the first request, records `expected_turn_count`, `turns_completed`, and `workload_complete`, and exposes `replayed_tokens` as the sum of server `suffix_tokens_dispatched` for each completed turn. Python syntax and `git diff --check` pass. These fields clarify artifacts; they do not establish persistence or correctness.
- LC-0009 CPU QSUM decode handoff was pushed separately at `f78949cf4f002c72a27de842ef869b6b177dc1b1`; the work confirms the needed raw one-token decode capture is absent, so LC-0009 remains open.
- Immediate follow-up was to push this checkpoint and sync the isolated Z840 worktree; completed at 01:27 below. GPU work remains gated on cooling confirmation.

## 01:27 follow-up

- Pushed coordinator commit `fa03664ccf021a2127f9f7110aa350ebc55e9977` and updated the isolated Z840 campaign worktree to that exact SHA; the worktree is clean.
- On Z840, the host-only production workload self-check passed again: exact initial request 65,536 tokens, seven source files, eight planned turns. The guarded-runner self-check also passed. Neither check accessed the GPU.
- The first self-check invocation used an incorrect build-directory path and did not launch; locating the existing binary resolved it without rebuilding. No source or host checkout was damaged.
- Next: hold GPU work for cooling confirmation and continue only host-side preparation or review.
- The clean Z840 campaign worktree was subsequently fast-forwarded to the latest docs-only checkpoint `e6fa5592770fb7bbc78954cf7acbf1387300b735`; it remains clean.

## 01:31 checkpoint

- Coordinator now contains the 64K-only suffix benchmark parity fix from suffix branch `60dfa`, integrated at `1b477ab`. The pinned ROCm 7.2.1 no-device build passed from `1b477ab`; log SHA-256 `0ded1e258db670cdc76718e921c9a28e2e420b1790a3ff0157514ec95fcecf58`, binary SHA-256 `34541d9e2f37c0598ad818f62b6a1d47c29161648893efab4a5e7a5d374c5c18`. No GPU benchmark ran.
- Production API streaming now requires `[DONE]` plus accepted `finish_reason`; its no-network self-check accepts a complete stream and rejects a truncated one. The workload atomically persists active turn/phase before each API request and saves returned request data immediately afterward. Local Python syntax and SSE self-checks pass. This does not qualify inference or prefix-reuse correctness.
- Cooling/airflow remains unconfirmed; no GPU work was started.

## 01:33 follow-up

- Pushed coordinator commit `d22915b` with the SSE completion gate, atomic in-progress request artifact, workload documentation, and 64K prefill build evidence.
- The clean isolated Z840 worktree fast-forwarded from `e6fa559` to `d22915b`. Remote Python syntax and the full tokenizer-backed `--self-check` passed: 65,536 initial ChatML tokens, seven source files, eight planned turns, SSE completion PASS, atomic artifact PASS. No GPU was accessed.

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
- The `miinfer` serving target built on the clean Z840 worktree (exit 0; binary SHA-256 `106feada3db19dca0d8b02bd9f5612e647b9175efabf3553ebc99a759d77e9a8`). A guarded startup at capacity 131,072 reached 34,254,819,328 B of 34,342,961,152 B VRAM and was stopped by the 1 GiB floor (wrapper exit 123) before the first 64K request. No prompt inference occurred.
- The current candidate uses bounded capacity 81,920, enough for the exact 65,536-token start and eight short turns within the observed MI50 memory budget. The serving whitelist extension is limited to this research-only value; final generation is capped at 256 tokens and each planned read_file call must match its path and line range. No 128K prompt is run.
- Review found that the initial workload summary recorded prefix counts but did not require reuse for later turns. The integrated script now requires each turn after the first to have both a nonzero reused prefix and a nonzero suffix dispatch; the overall command fails if any turn misses that gate.
- The 80K candidate, reduced tool-turn plan, and fail-closed planned-read check were pushed at `b940640`. On the clean Z840 worktree, `miinfer`, `miinfer-m31-accel-0001-agent64k`, and the host tokenizer counter rebuilt successfully. The updated host-only self-check counted exactly 65,536 initial tokens for eight turns; log `/mnt/lc9-lake/m31-accel-0001-captures/agent-host/native-self-check-81920.log`, SHA-256 `a7ad2173803692c76242352c56171899357f323901f7bcccd8010543f4e260a4`.
- First guarded production startup at 81,920 loaded successfully and peaked at 31,033,126,912 B VRAM / 30 C. It failed before inference because `git rev-parse HEAD` inside the container could not follow the mounted worktree's host-only `.git` pointer. The client traceback is retained in `agent64k-03/runner.log`; the host-verified source SHA is now passed into the container and pushed at `f01f7e9`.
- The next attempt, source `f01f7e9`, loaded the exact 65,536-token initial request at 81,920 context; server request 3 reached `queued`. It was stopped at the 85 C warning before any response: 31,056,838,656 B peak VRAM, 85 C junction, 100% GPU use, 147 W sampled, fan level 25% / 0 RPM. Exit 137 is intentional operator stop. Idle afterward: 38 C junction then 10,854,400 B VRAM and 0% GPU use. Capture `/mnt/lc9-lake/m31-accel-0001-captures/agent64k-04/`; environment/server/telemetry/guard SHA-256 values are in REPORT.md. `EXP-0353` records that setting fan control requires sudo and was denied to the runner. Hold sustained GPU work pending verified chassis airflow or cooling control; no multi-turn or performance result is established.

## 2026-10-12 01:35 checkpoint

- Z840 SSH is live; Lake is mounted read/write with 1.6 TB free. MI50 idle sample: 28 C junction, 0% use, approximately 10.9 MB VRAM; fan reports 15% / 0 RPM. Cooling remains unconfirmed; GPU jobs remain held.
- Decode worker `26b5f076e22ec29cdfd5609cacc7526ee2773b33` was reviewed and cherry-picked as coordinator `0ee18e610f25ce0ea1f5f4a82ea018c6bcbc082f`, pushed. The commit adds the production HIP Graph decode benchmark, CMake target, and instructions; it is a harness only, with no performance claim.
- Original Z840 campaign worktree remains clean at `d22915b`. A separate `/mnt/lc9-lake/m31-accel-0001-decode-build` worktree is detached at `0ee18e6` and building with the pinned ROCm 7.2.1 image, without GPU devices. Build/self-check result and artifact hashes are pending.
- Next: capture the actual Podman container logs and exit status, verify the benchmark self-check and binary hash, then update/push this evidence. Keep GPU execution gated on airflow confirmation.

## 2026-10-12 01:38 checkpoint

- The pinned ROCm 7.2.1 no-device build from `0ee18e610f25ce0ea1f5f4a82ea018c6bcbc082f` exited 0; `miinfer-m31-accel-0001-decode-bench --self-check` passed. Podman inspection shows `devices=[]`.
- Retained build log `/mnt/lc9-lake/m31-accel-0001-captures/build/decode-0ee18e6-build.log`, SHA-256 `2803889d8959cee46c61c6eee8f9244ee03efd32ec915334afeae05ed06e7f31`; binary SHA-256 `900729a294a61c1a564e7007d5dbf5a381b7f164aae7bd49cd6cdb40b96431b1`. Warnings were HIP compiler diagnostics for an unused parameter, unrolling, and occupancy.
- No decode timings were collected. The MI50 stayed idle; cooling/airflow is still unconfirmed and remains the blocker to GPU work.
- Next: obtain cooling confirmation, then schedule the guarded 64K workload before performance A/B runs. The benchmark is ready for that queue after cooling is cleared.

## 2026-10-12 01:40 checkpoint

- Synced the clean Z840 coordinator worktree from `d22915b` to pushed HEAD `c8c05c57a6902c2f99af41c4dbb5fe8a25f1feea`. It remains detached and clean; the original dirty checkout remains untouched.
- Separate Lake decode-build worktree remains at source `0ee18e6`, with the successful build, self-check, and retained log/binary hashes documented above.
- No GPU job was started. Cooling confirmation is still the hardware gate; once cleared, first priority is the guarded 64K production workload before decode/prefill A/B.

## 2026-10-12 01:42 checkpoint

- Fixed the production workload's fail-open reuse check: every request on turns 2–8 now requires both a nonzero reused prefix and nonzero suffix dispatch. Per-request checks and telemetry are saved into the result; absent telemetry fails closed. Commit `260b9ee60355887330869d96ad80d72f3a50660b` is pushed.
- Local `py_compile` and no-network `sse_self_check` passed. The clean Z840 campaign worktree was synced to `260b9ee`; its full tokenizer-backed `--self-check` passed exact 65,536-token counting, seven-file/eight-turn planning, SSE completion and atomic artifact checks. Log SHA-256: `64ee44bbfffe0cc57bb413a3970d353f1e97ecac59bdfaee5f86d90b36a53b25`. No GPU/server inference.
- Fan telemetry remains 0 RPM at idle. Cooling confirmation is still the GPU blocker; 64K multi-turn correctness and state continuity remain unqualified.

## 2026-10-12 01:44 checkpoint

- Deterministic parity setup for the focused suffix benchmark is pushed at `dcde94cf49f68af1e289045bba38f09dd5bb02f5`: cold, cached-prefix, and reuse generation now share greedy sampling, neutral penalties, and seed 42.
- Pinned ROCm 7.2.1 no-device build exited 0 from isolated `/mnt/lc9-lake/m31-accel-0001-prefix-build` at `dcde94c`; Podman reports `devices=[]`. Log SHA-256 `9edbf83be2bd3bf8156d54a3d68f2a364c8484fc90d0907733064ad1d32c1d96`; binary SHA-256 `0f9d89f488cc5ee3d9736862796de35bd6c664aa27f3a4c542bc6e4dd419a91f`.
- This only corrects a single-run parity gate. No suffix performance or state-continuity result exists; repeated paired samples and current-source profiling remain outstanding. Historical suffix attributions conflict across source revisions.
- GPU remains idle at 28 C, 0% use, 10.9 MB VRAM. Cooling/airflow remains unconfirmed. Next: after clearance, run the guarded 64K production workload before any suffix or decode A/B.

- The production API workload now requires exact reuse of the preceding request's full prompt-token prefix for every request after the first, plus a nonempty suffix dispatch. It records each expected/observed prefix count and fails on missing telemetry. Commit `529cf3cd84a6c1e5969fa785797d65d23683a74c` is pushed.
- Local Python compile and self-check passed; Z840 clean worktree synced to `529cf3c`, and tokenizer-backed workload `--self-check` again passed at exactly 65,536 tokens. No inference/GPU was run. Captured output hash is `64ee44bbfffe0cc57bb413a3970d353f1e97ecac59bdfaee5f86d90b36a53b25`.
- Next: clear cooling and run the guarded 64K production workload; exact-token prefix reuse is now an exit gate, but state/numerical continuity remains unqualified.

## 2026-10-12 01:46 live host recheck

- Z840 SSH works. Lake is mounted read/write with 1.6 TB free; root is 98% full with 11 GB free, so keep large builds and captures on Lake. No MIInfer/build process or tmux session is active.
- MI50 is idle at 28 C junction, 0% use, 10.9 MB VRAM, and 20 W. Both `rocm-smi` and `sensors` report fan 0 RPM; `ipmitool` is unavailable and no chassis-fan sensor is exposed. This cannot verify physical airflow, so the GPU gate remains closed.
- Existing isolated decode and suffix64 worktrees are clean at `26b5f07` and `1b477a`; suffix64 is already integrated and its no-device build passed. The campaign worktree remains clean at `529cf3c`. No new inference or performance evidence was produced.

## 2026-10-12 01:47 artifact revalidation

- Z840 campaign worktree remains clean at `529cf3c`. The exact-prefix host self-check log still matches SHA-256 `64ee44bbfffe0cc57bb413a3970d353f1e97ecac59bdfaee5f86d90b36a53b25` and records exit 0.
- The guarded 64K attempt's retained status is exit 137 with an explicit operator stop at 85 C, fan 0 RPM, 100% GPU use, and 147 W. Idle telemetry now reads 28 C, 0% use, and approximately 10.9 MB VRAM. This confirms the earlier safety stop; no GPU run was repeated.

## 2026-10-12 01:48 production telemetry gates

- The production workload now also fails closed unless server-reported `gqa_kv_reused_tokens` and `gdn_checkpoint_position` match the reused prefix, and `runtime_context_capacity` covers each prompt. The per-request artifact records these values and check results.
- Python syntax and no-network workload self-check passed. Self-check covers cold start, exact reuse, short prefix, empty suffix, incorrect GDN position, capacity overflow, missing telemetry, complete/truncated SSE, and atomic result writes.
- This gates reported positions and capacity, not numerical GDN/KV state equivalence. No GPU or tokenizer process was run; physical cooling remains unverified.

## 2026-10-12 01:49 Z840 tokenizer-backed validation

- Advanced the clean Lake campaign worktree to pushed source `4789d09`; `py_compile` and tokenizer-backed `--self-check` passed on Z840, counting exactly 65,536 initial ChatML tokens and exercising the position/capacity self-check. Log: `/mnt/lc9-lake/m31-accel-0001-captures/gates/position-capacity-4789d09-selfcheck.log`, SHA-256 `411305c5104f84aca5aff43517b13465de94fa3846a10a2b464896b94d627cbe`.
- The host worktree is clean at `4789d09`. This used CPU tokenization only; no inference or GPU work ran. Numerical state continuity remains unqualified pending a guarded multi-turn execution.

## 2026-10-12 01:50 GDN cursor correction candidate

- Source audit found that cold prefill passed `RecurrentLayerState` views by value and never advanced the storage cursor after forward. Eager recurrent decode reads that cursor for convolution indexing. The candidate now advances all 48 cursor values to the prompt end and after each consumed decode position, including stop/cancel paths; it reports min/max final positions.
- The production workload gates the observed GDN position range against each request's prompt and generated-token bounds. Local Python compilation and no-network self-check pass. C++ build and Z840 no-device build are pending; no GPU run is allowed until cooling is confirmed.
- This validates cursor bookkeeping, not numerical GDN/KV contents or exact cold/reuse output parity.

## 2026-10-12 01:55 no-device validation

- Pinned ROCm 7.2.1 container built `miinfer` and `miinfer-m31-accel-0001-agent64k` from `0941ae9` with no GPU devices passed. Exit 0; build log SHA-256 `ef06ee966407663b0caeb7607d8d1d34112e6b6c7e8f182647c9a48ef8ccdf59`.
- Native 64K harness self-check passed; binary SHA-256 `53f3bff20b6e0a57b19468d8bedb0aa28dc49b7ba51f3e72c0a5de84b00f1fd1`; self-check log SHA-256 `b51d580dbd526103240d11197dfdb2b6805567e71a65231bc84ebffd018cd9d9`.
- The tokenizer-backed production runner self-check also passed at 65,536 tokens from the same source; log SHA-256 `411305c5104f84aca5aff43517b13465de94fa3846a10a2b464896b94d627cbe`. GPU remained idle at 0% use and 10.9 MB VRAM. No inference ran.

## 2026-10-12 01:56 power-cap access check

- Checked whether lower power can substitute for unverified physical airflow. `sudo -n rocm-smi --setpoweroverdrive 100` is denied because this SSH account requires an interactive password. The cap was not changed; sysfs still reports the stock 225 W limit.
- MI50 remains idle at 28 C and 0 RPM. No GPU workload was launched. To proceed, either confirm chassis airflow across the passive card or set a temporary lower cap interactively, then I can run the guarded 64K workload and restore the device default afterward.

## 2026-10-12 01:58 request-metric attribution gate

- The production workload now requires cumulative prompt- and generated-token metric deltas to match the same server request record; absent or mistyped fields fail closed. Its self-check covers a matching record and a generated-token mismatch.
- Local Python compilation, no-network self-check, and `git diff --check` pass. This protects per-request decode/turn metrics from aggregate misattribution. Z840 revalidation is pending; no inference ran.

## 2026-10-12 01:59 Z840 runner verification

- The clean position-build worktree was advanced to `284435d`. Python compilation and tokenizer-backed `--self-check` passed on Z840, counting exactly 65,536 initial ChatML tokens with seven source files; the new metric-count mismatch check also passed in the no-network self-check.
- Retained log: `/mnt/lc9-lake/m31-accel-0001-captures/gates/request-metrics-284435d-selfcheck.log`, SHA-256 `411305c5104f84aca5aff43517b13465de94fa3846a10a2b464896b94d627cbe`. No server request or GPU inference ran.

## 2026-10-12 02:05 paired 64K suffix benchmark

- Commit `a8fe7c1cb7ac1de0e16e95980fdd782faa772259` is pushed. The focused 64K prefix-reuse benchmark warms suffix graph capture, runs five alternating cold/reuse pairs, checks exact generated-token and stop-reason parity for every pair, and reports per-pair TTFT plus median/observed range for TTFT and decode step time. Its CPU-only `--self-check` covers the five-sample count, median, and order schedule.
- Z840 built `miinfer-prefix-state-reuse-bench` from the exact pushed source in `localhost/miinfer-dev:rocm-7.2.1` (`sha256:bdc5ed42c985a6a333083e023225f248825a6e1511f38b6b50fbc7d5ace3fe9e`). No GPU device was passed. Build exit 0; self-check passed. Build log SHA-256 `4763a05116376334618023849cbc100167c9813d2c5c3e98beddb925b9b926a1`; binary SHA-256 `49654be84fe5e4dc25a148767ddb4b460e33f4036ba9c7840cf7c1e09dc0ef15`.
- The Lake disk is mounted read/write with 1.6 TB free. Post-build MI50 telemetry remained idle at 27 C junction, 0% use, and 10.9 MB VRAM. Its fan reports 15% / 0 RPM; physical airflow is still unconfirmed. This validates the benchmark build and self-check only; no suffix performance or 64K inference result exists. Next: after cooling clearance, prioritize the guarded production 64K workload.

## 2026-10-12 02:08 guarded suffix launch

- Commit `7c50a7ac5d5621232f2580781f42ef2cdd4f95a5` adds `prefix64k` mode to `run-guarded.sh`, invoking the five-pair 64K suffix benchmark under retained temperature/VRAM monitoring. Workload mode validation happens before hardware preflight. The self-check now verifies accepted/rejected modes as well as thermal and VRAM thresholds.
- `bash -n` and the guard self-check passed locally and on Z840 at this exact source. No GPU device or inference was used. The live 64K production workload remains first in the queue after cooling clearance; fan is still 15% / 0 RPM and physical airflow is unconfirmed.

## 2026-10-12 02:10 guarded-path build

- Built the benchmark at the exact `build/m31-accel-0001/miinfer-prefix-state-reuse-bench` path expected by `prefix64k`, from pushed source `f621596f8c721ea0699f98d80514eebb80e52438`, in the pinned ROCm 7.2.1 image with no GPU device passed. Build and `--self-check` exited 0. Build log SHA-256 `91a0f50e8973244eb7f95c650cbff79e8edfa12b999d449a50f6a120cac528ff`; binary SHA-256 `94b155e2b30f321c32f209e19e3a4ba16132ab146b53e8ad7694a562d36a5a7f`.
- Z840 remained idle afterward at 27 C junction, 0% use, 10.9 MB VRAM. The guarded suffix command is now build-ready, but no inference was run and cooling still blocks GPU execution.

## 2026-10-12 02:14 Machinist CPU-only portability build

- The existing Machinist `Development/MIInfer` checkout has unrelated dirty M30 changes and was left untouched. A separate clean checkout at `00f0d8e24228bda204ed187f227035ee1b368bd7` built the prefix benchmark with no GPU devices passed to Podman; build exit 0 and `--self-check` passed. Build log SHA-256 `ba572163f04ed20dbd0d7fa7a8e0c1fb14d7691adf2f5798cc59699403fd6c12`.
- Its binary SHA-256 matches Z840 exactly: `94b155e2b30f321c32f209e19e3a4ba16132ab146b53e8ad7694a562d36a5a7f`. Machinist's image digest is `sha256:447907c697fb2c39ce9657401e354a70753d5672ad955d4ceb12de976b91224a`, different from Z840's pinned image; treat this only as a cross-host compile check. No Machinist GPU or H→D operation ran.

## 2026-10-12 02:18 final benchmark-source rebuild

- The timing output now labels decode step values as `ms/token`. Source commit `fea33de7eb0a2a0d1582d297c607692ee308cb66` rebuilt on Z840 in the pinned image and passed `--self-check`. Build log SHA-256 `31c73bf0d8237bd3b0898c598079e301fa46df848244c4a2cf2c99fb0d8cc684`; binary SHA-256 `24d075003138f068f66c9a9273b4f08af51ee361b7db5aa568a827ab3e518a43`.
- Machinist rebuilt the exact benchmark source file (SHA-256 `c880be8ae7cd06024e7bd5fd488f2200ac31cf906add257edbf577297f9adb94`) on clean base commit `00f0d8e24228bda204ed187f227035ee1b368bd7`; build and self-check passed. Its log and binary hashes match Z840 exactly despite the local image digest difference. No device was passed to either build. Z840 remains idle at 27 C, 0% use; airflow remains unconfirmed.

## 2026-10-12 02:19 production workload revalidation

- At pushed source `974c17343d0b456822039ef18cd2a7e711978ada`, Z840 rebuilt `miinfer-m31-accel-0001-agent64k` in the pinned image with no device passed; `--self-check` passed (eight turns, 65,536 prompt tokens). Build log SHA-256 `501b5eb5318122d5cafc063807627acc7f32c3bfe65daf8af878854f4299e6db`; binary SHA-256 `2afaac0973ffd437d6be1c14cb6c44cf98a50fcb81d8326444af4f8024d19f83`.
- Built `miinfer-m31-agent-token-count` and ran the production workload `--self-check` using the pinned model mounted read-only. First try failed because SELinux labeling denied the container read; adding the existing guard's `--security-opt=label=disable` fixed it. The retry passed at exactly 65,536 initial ChatML tokens with seven source files, SSE completion, and atomic artifact checks. Tokenizer binary SHA-256 `66ca05ebd5d5932991e3fd6f88976d800b001dc817d128c57f2b710007d96b42`; retained retry log SHA-256 `411305c5104f84aca5aff43517b13465de94fa3846a10a2b464896b94d627cbe`.
- No API/server request or GPU device was used. Idle MI50 remains at 27 C, 0% utilization, and 10.9 MB VRAM. This verifies workload construction and harness checks only; multi-turn persistence remains unqualified pending a safe live run.

## 2026-10-12 02:20 persistent exit-marker audit

- Found an error in the detached tmux commands used for CPU-only builds: `$?` was expanded by the launching shell before tmux started, so their `.exit` marker files are not authoritative. The captured build logs themselves show completed targets and explicit self-check PASS output; the first tokenizer failure has a clear model-open traceback, and its misleading marker was not used to classify it as a pass.
- Verified corrected `\$?` escaping with a deliberate `false` probe: its retained marker contained `1` and the tmux session exited. Future persistent jobs will use the corrected capture or the tracked guarded runner. Probe marker: `/mnt/lc9-lake/m31-accel-0001-captures/build/exit-capture-probe.txt`.
- Reconciled Z840's Lake worktree to current pushed coordinator `a1f84e3`; it is clean and detached. No MIInfer process or tmux session is active. The MI50 is idle at 28 C, 0% use, and 10.9 MB VRAM; cooling/airflow remains unconfirmed.

## 2026-10-12 02:26 remote recovery and QSUM checker

- Z840 SSH is available. Lake is mounted read/write with 1.6 TB free; `/home` has 11 GB free. No MIInfer/GPU workload or tmux session was active. MI50 is idle at 27 C junction, 0% use, 20 W, and 10.9 MB VRAM. Fan telemetry remains 15% / 0 RPM, so physical airflow is unconfirmed and GPU inference stays paused.
- Verified the clean detached Lake source worktree at `a1f84e3180a93f416ef72621df3a5e40cba46ac8`; the dirty `~/Development/mi50` checkout remains untouched.
- LC-0009 already has a one-shot eager decode trace hook for post-normalized inputs, Q8 blocks, and Gate/Up outputs. The isolated branch `experiment/m31-lc-0009-qsum-checker` (`c00136d00c4e5a00c4b5460a1e31f793d4fd435e`) adds the missing checker QSUM mode and corrects the handoff. `--self-check` passed, and the default ordinary mode reproduced the Lake reference result: 160/160 Q8 groups and 17,408/17,408 rows each for Gate and Up within bounds. Branch pushed. No QSUM GPU capture was run; LC-0009 remains open.
- Decode/prefill timings and the 64K persistent-agent workload remain unmeasured on GPU. Next: continue source/build review and remote CPU checks; resume one guarded MI50 run only after airflow or equivalent cooling safety is confirmed.

## 2026-10-12 02:34 Z840 no-device build and self-checks

- Z840 SSH recovered. The clean Lake build worktree is pinned to source `bf5f6d0bd991939bf23ac5af6878350c6ac9621b`; the build completed with exit 0 in the pinned ROCm 7.2.1 image without passing a GPU device. It built `miinfer`, the 64K harness, tokenizer counter, prefix reuse benchmark, and decode benchmark. Build log SHA-256 `86272bd30fd1b3f7102a1658b9bb3ffaeeca962e7635725fd059fcac19c82e68`; exit marker SHA-256 `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`.
- Native self-checks passed: 64K harness (8 turns, 65,536 prompt tokens), prefix benchmark (5 alternating pairs), and decode benchmark (deterministic prompt and bounds). The exact pushed production workload script from `e145b71` (SHA-256 `1ba6d46abc7f871d16779927d9c0cfe5ee0f0dd9562a549592dac53a859966fb`) passed the tokenizer-backed self-check at exactly 65,536 prompt tokens, with seven source files, SSE completion, and atomic artifact checks. These validate build and harness construction only; no API request, inference, or GPU timing ran. Binaries and build log are retained under `/mnt/lc9-lake/m31-accel-0001-cursor-build/` and `...-log/`.
- The 64K cursor fix is built from `bf5f6d0`; the first-request cold-start fail-closed gate at coordinator `e145b71` was included in the production workload script and passed its local and tokenizer-backed self-checks. MI50 was idle at 27 C junction and 0% use; telemetry still showed fan 15% / 0 RPM, so airflow remains unconfirmed and GPU execution stays paused.
- Next: confirm physical airflow/cooling, then run the guarded production 64K multi-turn correctness workload first. Until then, do not claim GDN/KV numerical continuity, agent reliability, or performance results.

## 2026-10-12 02:41 decode_step cursor fix

- A fresh call-site audit found `PrefillV2Model::decode_step()` failed to synchronize recurrent storage positions around the GDN decode calls. Unlike production `generate()`, GDN receives position through a by-value view, while GQA uses the explicit argument. Repeated calls could therefore use a stale GDN convolution cursor. The fix is pushed in `b06bffd648bfb8258dc115b7584b1842cfb03883`: set all 48 cursors to the supplied position before dispatch, then to `position + 1` after the recurrent blocks. Callers include `prefill_v2_e2e_bench.cpp`, `v2_0045_qualification_bench.cpp`, and `decode_stage_profile.cpp`.
- The clean Lake build worktree was advanced to `b06bffd`; pinned ROCm 7.2.1 no-device build exited 0. Native 64K, prefix, and decode harness self-checks passed; the exact 65,536-token production workload self-check also passed (7 source files, SSE completion, atomic artifact). Build log SHA-256 `dc52e1f4b87042f885f3fa4ceea4656136dd46f84ec1ce359b8f126125a619fc`; check log SHA-256 `71b47d65bbb1057d8eadc683f8b0b8b75c10c08eae83d1b6956638b1cf2aa3ea`.
- These are compile and harness checks, not a live GDN position test. The MI50 remained idle at 27 C and 0% use, with fan telemetry 15% / 0 RPM. Cooling/airflow remains unconfirmed. No GPU inference or timing ran.
- Source/profile review found no current-source decode or prefill measurements to justify a kernel optimization. Next: confirm cooling, run guarded production 64K correctness first, then the ready five-pair cold-versus-64K-prefix/512-suffix benchmark. Decode/prefill tuning follows current-source profiling; no default changes are justified yet.

## 2026-10-12 02:51 Machinist CPU build

- The dirty `/home/machinist/Development/MIInfer` checkout remains untouched. A separate clean HTTPS clone at `/home/machinist/Development/MIInfer-m31-b06bffd` is detached at `b06bffd648bfb8258dc115b7584b1842cfb03883`.
- Machinist's `localhost/miinfer-dev:rocm-7.2.1` image (`sha256:50ff28a28cc13a3ca7333cd0c70e322d16245967c5fdf0d30a1bf57bda80a512`) built `miinfer` and the native 64K harness with no GPU device passed; exit 0. The 64K harness self-check and thermal-guard self-check passed. Build log SHA-256 `9be32e63d2d174343848e1d04cda603c081159914348ac54b88411462635f4cc`; check log SHA-256 `a44a04e89101059c71c8f2a4452428d316488a894bfef023e3339d49f79b0b30`.
- Machinist reports two adapters; target MI50 GPU[1] remained at 0% utilization and 0% VRAM use. No GPU was passed to Podman and no GPU workload ran. This is cross-host compile validation only, not MI50 qualification. Z840 cooling remains unconfirmed; the 64K production GPU run is still held.

## 2026-10-12 02:53 Machinist host-only tests

- On the same clean `b06bffd` clone, built the registered host-test targets and ran CTest `-L host-only` in the no-device ROCm container. All 13/13 host-only tests passed (1.34 s total); exit marker 0. Log SHA-256 `48b95d08352ddb7a50cc35884f737800eeb089a9d2382454bafcd834ed79a714`.
- These are repository host regressions and do not exercise MI50 GDN cursor behavior, GPU kernels, or 64K numerical state continuity. No GPU work ran.

## 2026-10-12 02:54 cooling telemetry interpretation

- AMD's MI50 product specification lists the board as passively cooled with 300 W peak TDP ([AMD MI50 specifications](https://www.amd.com/en/support/downloads/drivers.html/accelerators/instinct/instinct-mi-series/instinct-mi50-32gb.html)). Therefore the card's 0 RPM fan sensor does not tell us whether chassis airflow is adequate. Physical airflow through the passive heatsink remains unconfirmed; the prior 85 C operator stop remains in force until that is clarified.

## 2026-10-12 02:56 SSH recovery and runner correction

- Z840 SSH succeeded; Lake is mounted read/write with 1.6 TB free, while `/home` has 11 GB free. MI50 is idle at 27 C, 0% use, 20 W and 10.9 MB VRAM; fan telemetry remains 15% / 0 RPM. Physical heatsink airflow remains unconfirmed, so no GPU work ran.
- The stopped `agent64k-04` directory has no workload JSON or saved prompt manifest/hash. Its environment identifies run source `f01f7e9a210489139d7aa36e4391ede48e723464`; the server log shows request 3 queued and operator stop at 85 C, container exit 137. No response was completed.
- Coordinator commit `0e9d6123f22d157db7a97bb250b04f405ee9591b` independently tokenizes each actual API payload and checks the result against per-request server token telemetry. Its per-turn aggregate decode rate now time-weights server `decode_tokens_per_second` by `decode_ms`, keeping the same token/time basis. Python compilation and no-network runner self-check passed locally; `git diff --check` passed before this report edit. Z840 tokenizer-backed validation of this newest script is pending; no inference or GPU request was made.
- Decode/prefill A/B results and completed production 64K qualification remain absent. Next: confirm passive-card airflow, sync/build the current source, run guarded 64K correctness, then run the prepared five-pair cold/suffix prefill comparison.

## 2026-10-12 02:58 Z840 runner validation

- Fast-forwarded the clean Lake build worktree from `b06bffd` to pushed coordinator HEAD `07a8489731421f70d768074f03e76fe062b67196`. Python compilation and the tokenizer-backed production workload `--self-check` passed: exactly 65,536 initial prompt tokens, eight turns, seven source files, SSE completion, and atomic artifact checks. Retained log SHA-256 `411305c5104f84aca5aff43517b13465de94fa3846a10a2b464896b94d627cbe`.
- The MI50 remained idle at 27 C junction, 0% use, 20 W and 0% VRAM use; fan telemetry remains 15% / 0 RPM. No server request or inference ran. This validates prompt construction and runner self-checks only; the 64K production workload, state continuity, and all performance results remain unqualified pending airflow confirmation.

## 2026-10-12 03:00 replayable request artifacts

- Pushed coordinator commit `a9480ae062bb83def6bb2f60e3b68a74b377bf9b`. The production runner now atomically saves each exact API request body beside the JSON result before dispatch and records artifact path, SHA-256, byte count, and MIInfer-tokenizer prompt count. Result metadata captures the source SHA, runner SHA, and the model SHA passed by the guarded launcher. Caught request errors mark the partial result `failed` while retaining active phase/request identity.
- Local Python compilation, runner self-check (artifact round-trip and failure record included), and `run-guarded.sh --self-check` passed. On Z840, the clean Lake worktree fast-forwarded to `a9480ae`; tokenizer-backed workload self-check passed at 65,536 tokens/eight turns/seven files, with log SHA-256 `411305c5104f84aca5aff43517b13465de94fa3846a10a2b464896b94d627cbe`.
- The MI50 remained idle at 27 C and 0% use. No API request or inference ran. These artifact safeguards improve the next run's evidence; the previous `agent64k-04` capture still lacks a workload JSON and replayable prompt. Physical airflow remains unconfirmed, and 64K state correctness and performance remain unqualified.

## 2026-10-12 03:01 Machinist runner validation

- The clean `/home/machinist/Development/MIInfer-m31-b06bffd` clone was fast-forwarded to the exact runner source `a9480ae062bb83def6bb2f60e3b68a74b377bf9b`. Its no-device ROCm container built `miinfer-m31-agent-token-count`; Python compilation and tokenizer-backed workload self-check passed at 65,536 tokens/eight turns/seven source files. Log SHA-256 `05d50d96d701751a74ed75a60cb9a229f36ea7e44944de1e47f3836439566067`.
- No GPU was passed to Podman and no server request ran. Together with Z840's matching check, this verifies the replayable input artifact path on both hosts, but does not prove GPU context continuity, inference correctness, or performance.
