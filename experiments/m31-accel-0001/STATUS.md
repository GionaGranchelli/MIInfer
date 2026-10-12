# M31-ACCEL-0001 live status

Updated: 2026-10-12 06:54 Europe/Amsterdam
Deadline: 2026-10-12 10:00 Europe/Amsterdam

## 2026-10-12 06:08 Guard BDF binding pushed and synced

- A static acceptance audit found the guard sampled the first unscoped `rocm-smi` GPU row while the container selected runtime device 0. Fresh Z840 inventory shows its only ROCm device is GPU[0], BDF `0000:06:00.0`, MI50/gfx906; no mis-targeted run was observed.
- Updated `run-guarded.sh` to require exactly one ROCm device at the qualified BDF, resolve its `rocm-smi` index, query telemetry only for that index, pass the resolved index to `ROCR_VISIBLE_DEVICES`, and record BDF/index in `environment.txt`. It refuses absent, wrong, or ambiguous inventories before model inference. Commit `0477436f13c443fed89a11a1131e0e7d799049e9` is pushed and remote-verified.
- The no-device guard self-check tests correct BDF, wrong BDF, and ambiguous multi-device inventory. `bash -n`, guard self-check, and `git diff --check` passed locally; host `bash -n` and guard self-check also passed after syncing the clean Lake worktree to `0477436`.
- Live `rocm-smi --showbus --showproductname` confirmed a single GPU[0] at `0000:06:00.0`, MI50/gfx906. `rocm-smi -d 0 --showtemp --showmeminfo vram` read that device at 25 C junction and 10.9 MB VRAM. All five staged binary hashes remain unchanged. No inference ran.
- Machinist is reachable and idle. Its primary checkout at `d8b8336` is dirty and was left untouched; no persistent jobs or containers were present. Its available RAM/disk and branch status are documented in this update's evidence.
- Airflow remains unconfirmed; no GPU run is allowed yet. After cooling clearance, run the guarded production 64K attempt first.

## 2026-10-12 06:00 Z840 launch worktree synchronized

- Updated only the clean Lake worktree `/mnt/lc9-lake/m31-accel-0001-integrated` to the pushed coordinator SHA `630ce335da474f321f0f529f66ca1039003a829d`. Direct Z840-to-GitHub HTTPS fetch was stuck in `SYN-SENT`; stopped that fetch and transferred a verified 9 KB bundle over SSH. The user's dirty primary checkout was untouched.
- On Z840, `bash -n` for both launch scripts and `run-guarded.sh --self-check` passed at the integrated SHA. All five staged binary hashes match the earlier pinned build outputs.
- Ran the integrated production runner self-check in the pinned container with `--network=none`, model mounted read-only, and no GPU devices. It passed and confirmed exactly 65,536 prompt tokens and 65,536 token IDs.
- No GPU inference or performance sample ran. Physical airflow remains unconfirmed; after clearance, the prepared production 64K run is first in the queue.

## 2026-10-12 05:52 Z840 preflight and guarded-VRAM integration

- Z840 SSH succeeded. Lake is mounted read/write at `/mnt/lc9-lake` with 1.6 TB free; `/home` has 11 GB free. MI50 is idle at 26 C junction, 0% use, with 10.9 MB VRAM sampled and no active Podman containers. No GPU workload was started.
- Reviewed the pushed VRAM-window candidate and its callers. Host checks passed: Python compilation, agent workload self-check, structured telemetry window/peak self-check, both shell syntax checks, guard self-check, and `git diff --check`.
- Integrated candidate `543629d31581ef39e788180b1b9742d30ea0d273` as coordinator commit `f8974d0`. Per-request sampled VRAM now includes guard samples inside the request time window plus boundary snapshots; it remains a sampled maximum, not a continuous peak.
- Physical airflow through the passive MI50 heatsink is still unconfirmed. Keep guarded GPU work held until airflow is verified. Next after clearance: run the already-built production 64K workload, then the matched eager GQA A/B.

## 2026-10-12 05:17 GQA eager-route benchmark repair

- The route-aware harness is pushed at `dcd51cb1ae732f9c5dbd6da623b82a5873b90b37` (`experiment/m31-accel-0001-gqa-eager-route`). It selects eager or HIP Graph explicitly, rejects route mismatches, records the route, and allows the existing thermal guard to validate/run a clean external source worktree.
- Applied the same benchmark-only route change to the GQA A/B branches; runtime bases remain the approved control/candidate. Both branches are published and `ls-remote` verified: control `16662fcbc22c478e897705a3eb6c59ee390fd88c`; candidate `527baba9f1f3915a6dc2507d1b8973089fb54827`.
- Both route-enabled targets rebuilt in pinned image ID `50ff28a28cc13a3ca7333cd0c70e322d16245967c5fdf0d30a1bf57bda80a512`, with `--network=none` and no GPU device nodes. Control/candidate self-checks passed. Build metadata embeds exact clean source SHAs `16662fcbc22c` / `527baba9f1f`; binary hashes `ed631a655d2c4c802de026ef0cdde705b7ee583ed8334fefa84d956fe2df3c08` / `026ff260630d4a7e63a3ea4d02ead847699d4cf31191d4d3d6805ef2ce7097a9`. Log hashes: `129a859cdf0495cd2ce0a0feca3f4433bf1ad8e8988ec590fb079e2b5e7da472` / `707bee05ba2ced1a83f6b6f34d9c7249e1964e53ca97c8c1e28f55e2c9f14781`.
- The local ROCm 6.2 configure failed because `lld` could not load `libxml2.so.2`; the pinned Z840 ROCm 7.2.1 no-device build succeeded. Z840 GitHub SSH authentication is unavailable, so the updated branch commits were transferred as thin bundles and pushed through coordinator HTTPS.
- No GPU sample ran. The physical airflow confirmation remains pending; `sudo -n modprobe hp_wmi_sensors` was denied, so chassis fan telemetry remains unavailable. After airflow is confirmed, use the exact retained production prompt artifact, run both branches in eager mode under the guard, and reject timing comparisons if generated histories differ.

## 2026-10-12 05:09 GQA A/B validity audit

- The published A/B worktrees and no-device build/self-check evidence are real, but the current benchmark is invalid for the intended GQA change: candidate `8a9caec` modifies the eager decode path, while the harness unconditionally requests HIP Graph and rejects eager samples. Do not use these captures to claim a GQA result. A route-selectable harness and guarded external-worktree handoff are being prepared.
- The stopped production run retained a valid shared 65,536-token prompt artifact at `/mnt/lc9-lake/m31-accel-0001-captures/agent64k-05/agent-workload.inputs/turn-01-request-01.tokens.json`. Verified model SHA matches the pinned model; token count is exactly 65,536 and recomputed token-ID SHA-256 matches `e95d256169a1882a8708d7c511d271d03edc4973e912111284844be4a533bf5d`. No second-request history exists because the run stopped before responding.
- Latest host preflight: Z840 and Machinist are reachable and idle; MI50 is 26 C junction, 0% busy, and no process has `/dev/kfd` open. Physical airflow remains unconfirmed, so no GPU workload is allowed yet.

## 2026-10-12 05:02 SSH restored and GQA A/B branches published

- Z840 SSH succeeded. Lake is mounted at `/mnt/lc9-lake` with 1.6 TB free. The MI50 reports 25 C, 0% use, and 10.9 MB VRAM; no inference was started. `/home/fedora-workstation/Development/mi50` is still dirty and untouched.
- Fresh ROCm SMI reports 25 C edge / 26 C junction, 0% use, and 0 RPM. No process has `/dev/kfd` open. A `pgrep` match was the retained tmux wrapper for `agent64k-05`; inspecting the pane confirms it is dead with guard exit 122 at 04:04:41, not an active run.
- Published the prepared benchmark-overlay branches to `origin`: control `7ee6b3dedf16c1f3a25794ac46e3779e6f91b2cb` and candidate `7bcca14a205dab71923b3d24611db162172f3eb5`. `ls-remote` verified both exact SHAs. Each branch uses the same benchmark source overlay on its approved runtime base; both pinned ROCm no-device builds and self-checks passed.
- Direct push from Z840 is unavailable because its GitHub SSH key is rejected (`Permission denied (publickey)`). Transferred only the two branch commits with a small Git bundle whose required base commits were already present locally, then pushed through the coordinator's HTTPS remote.
- Airflow through the MI50 passive heatsink remains physically unconfirmed. Keep the GPU gate closed. After cooling is verified, run a fresh guarded 64K correctness capture first; then run the matched control/candidate A/B. Existing 85 C-stopped capture is terminal. Correctness and performance remain unqualified.

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

## 2026-10-12 03:13 SSH restored

- Z840 SSH succeeded twice. Lake is mounted read/write at `/mnt/lc9-lake` with 1.6 TB free; `/` has 11 GB free. No MIInfer, campaign runner, guarded benchmark, or matching tmux process is active.
- MI50 is idle at 27 C junction, 0% use, 20 W, and 10.9 MB VRAM. Sysfs exposes only the MI50's own fan input (`0 RPM`) and an HP WMI PWM enable value; no chassis fan RPM is available. The separate `hp_wmi_sensors` driver is installed but unloaded. Loading it was not attempted; even rotor telemetry would not prove airflow through the passive heatsink.
- `/home/fedora-workstation/Development/mi50` remains dirty with M30 changes and artifacts; it was not touched. No GPU work ran. Cooling clearance remains the gate before guarded 64K inference or timing.

## 2026-10-12 03:21 benchmark gates tightened

- Pushed coordinator `38eccc295b39e8c9db35ea1ab78c13ad870c9855`. The production runner now rejects reuse telemetry unless `reused_prefix_tokens` is within the prompt and reused plus dispatched suffix tokens equals the prompt length. Its no-network SSE/telemetry self-check covers a dispatch gap and passes; `py_compile` passes.
- The 64K prefix benchmark now rejects cold/reuse samples without HIP Graph or with non-finite/non-positive prefill, TTFT, decode, or total timings; restore/suffix timings must be finite and nonnegative. Its self-check covers graph fallback and NaN TTFT and passes.
- Machinist built `miinfer-prefix-state-reuse-bench` from exact source `38eccc2` in the pinned ROCm 7.2.1 image with no device passed, then `--self-check` passed. Binary SHA-256 `e71c1f3720a4c189568d22e1de303284d381a48c6457f024c01296b6b1e2cd98`; C++ source SHA-256 `d939692d9d3accded76709095b557b4c625706111c023e786bd3ea826ce4f931`. First fresh-worktree configure hit DNS failure while fetching nlohmann/json; retry reused the existing build dependency cache with network disabled and succeeded. Temporary review worktree: `/tmp/m31-accel-0001-38eccc2-review` on Machinist.
- Review found the prefix and decode fixtures are synthetic token IDs. These harnesses can characterize the model execution paths but cannot stand in for measurements on the actual coding-agent prompt. No GPU work or live performance measurement ran; airflow remains unconfirmed.

## 2026-10-12 03:22 decode benchmark gate

- Pushed `27c3ab3`; the decode benchmark now requires HIP Graph, finite positive TTFT, decode time, and decode throughput on every sample. Its self-check rejects graph fallback and infinite decode time.
- Machinist rebuilt `miinfer-m31-accel-0001-decode-bench` from exact source `27c3ab3` in the cached pinned ROCm 7.2.1 build with no device passed. `--self-check` passed. Binary SHA-256 `3fa0b094cf0bb06539e66c59ff7343a32f9a4497d927e8a73d33721cf4eb095a`; source SHA-256 `615c480819a7a68c951732844273df90495a40878d3b37b5e1e66d30365b6463`.
- No inference or performance sample ran. Decode remains synthetic-input evidence until run on a workload-derived token history; 64K correctness and cooling remain open.

## 2026-10-12 03:24 guarded decode launch

- `run-guarded.sh` now accepts `decode` mode and launches the compiled production-path benchmark with defaults P=65,536, TG=128, and five samples. It writes `/captures/decode.json` while preserving the existing temperature/VRAM log and stop checks. `bench/README.md` documents baseline/candidate capture commands and matching requirements.
- Local `bash -n`, guard `--self-check`, and `git diff --check` passed. No inference ran. Exact Z840 source sync/build remains pending; airflow confirmation still gates all GPU work.

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

## 2026-10-12 03:25 Z840 decode launch-path build

- SSH succeeded; Lake is mounted read/write with 1.6 TB free. The clean remote worktree is at pushed `c30bcdeb8081fb0d7832d4ecc78b1df2e16fb7c3`. Pinned ROCm 7.2.1 no-device build log SHA-256: `942192a2218f2e9247c659e0e29ac7c2cbcfc0f5304369d8eefc12848f5e560a`. Prefix and decode benchmark binary SHA-256: `e71c1f3720a4c189568d22e1de303284d381a48c6457f024c01296b6b1e2cd98` and `3fa0b094cf0bb06539e66c59ff7343a32f9a4497d927e8a73d33721cf4eb095a`. Prefix, decode, and guard self-checks pass.
- The MI50 is idle at 27 C, 0% use, and 0% VRAM use; fan telemetry remains 15% / 0 RPM. This is build/harness evidence only. Physical airflow is unconfirmed; no inference or performance run was made.

## 2026-10-12 03:27 production token-ID artifacts

- Pushed coordinator commit `8bd8b4c6e7dae14379f921101237c8a513df61e7` adds an explicit token-ID mode to the existing ChatML tokenizer helper. The production workload now saves each request's exact token IDs with the request-body SHA-256, model ID, token count, and canonical little-endian token-ID SHA-256. It checks the initial token-ID count against the exact 65,536 count. Python compilation and diff checks passed locally.
- Z840 rebuilt `miinfer-m31-agent-token-count` in the pinned ROCm 7.2.1 image with no GPU device passed. Build log SHA-256 `d2f438121df46da2bd1755b044e64f2eddf33c74974975211a7e411f3fae0243`; binary SHA-256 `900f5dfafc886d7f5e9f2d6389634bda92aa1e9cac0f33c6db5b1d3316b10614`. The tokenizer-backed no-GPU self-check passed at exactly 65,536 prompt tokens and 65,536 emitted IDs; log SHA-256 `b8e9f5e3c31b9868479408d706dd1abdd2fdd9f3f65bf79e9dec4e7b1a00c2c2`.
- These artifacts allow real request token sequences to be paired and their LCP measured, but the current prefix/decode benchmark fixtures still use synthetic IDs and do not yet consume the artifact. No live request or GPU work ran; passive-card airflow remains unconfirmed.

## 2026-10-12 03:31 production-token benchmark build

- Pushed source `ff4ff7343822ccea1fd12b512f8393e9fc778a51` adds validated `.tokens.json` loading to both benchmarks. Decode accepts one exact production prompt; prefix reuse accepts adjacent request artifacts and derives their longest common token prefix, retaining the original cold-vs-reuse parity and timing gates. The shared loader checks schema, model label, request hash shape, token count, token hash, integer range, and vocabulary bounds.
- Z840 built both targets in the pinned ROCm 7.2.1 container with no GPU passed. Build log SHA-256 `ad4487ab5c13cbd435a7e6e772b25e640b8ebcdc0068f1dd3c69b15b57d0541e`; decode and prefix binary SHA-256 `5f53e5bac80c9aac82b04593de72c4ef40b00c349a2038a7ba732ff8c4d6364d` and `a5f21b0bf206d0e8e8ad952ffdca2a17e5ea7862481848d163b32aa27ce3b43e`. Both `--self-check` runs pass; combined log SHA-256 `0b61087d07dbc3b5fc6412d73e5e946fb04ae7bae41def89df410634be9c38c0`.
- The guarded launcher now accepts artifact filenames staged in a new capture directory and rejects missing or path-containing filenames before starting the GPU container. Shell syntax and guard self-check pass locally. Z840 worktree is clean at `ff4ff73`; idle MI50 was 26 C, 0% use, 0% VRAM, fan 15% / 0 RPM. Physical airflow remains unconfirmed; no inference or timing run was made.
- Next hardware step remains confirming airflow through the passive heatsink. After that, run the guarded exact 64K agent capture, then stage its saved request token artifacts for decode and consecutive-request prefix A/B. The benchmarks are now prepared for real production token IDs, but no such captured request pair exists yet.

## 2026-10-12 03:39 SSH recovery and f271 regression check

- Both hosts are reachable again and their clean campaign checkouts are at pushed source `f2715b246256b93883176ee90e17935595f041d4`. Lake is mounted read/write with 1.6 TB free. No GPU run was started on either host.
- Machinist built the registered host-only regression targets and passed CTest `-L host-only`: 13/13 passed in 1.33 s using the existing dependency-populated build directory, inside the pinned ROCm container with no device nodes passed. Log: `/home/machinist/Development/MIInfer-m31-captures/f2715b2-host-regression.log`, SHA-256 `8e096862ea5c53d758177b8c08cf63113fac65bcb389293a028def1a851ca20c`.
- A separate fresh offline CMake build directory could not fetch `nlohmann/json` because the container had no DNS/network; this did not affect the existing build or checkout. Reusing its populated dependency cache completed the checks. Reviewer recheck of `f2715b2` found the workload/model provenance and exact-prefix validation findings resolved.
- This is host regression evidence only: no GPU correctness, live 64K session, or performance A/B result was produced. Passive-heatsink airflow is still unconfirmed, so Z840 GPU work remains held. Next: obtain airflow confirmation, then run the guarded 64K correctness workload before any performance captures.

## 2026-10-12 03:48 guard safety and acceptance audit

- Read-only acceptance audit of pushed `1c80856` found that the detached GPU container could outlive the SSH-side guard monitor, and the 85 C warning only logged while allowing the workload to continue. The monitor now stops the run at 85 C pending operator inspection, stops its container on HUP/INT/TERM, and self-checks the temperature action at 84.9/85/90 C.
- The guard now refuses a dirty source worktree, verifies the mode-specific executable files are present, and records SHA-256 for every executable used. The documented Z840 command launches it in tmux and records the pane PID, so SSH loss does not remove the monitor.
- `bash -n`, `run-guarded.sh --self-check`, and `git diff --check` pass. No GPU workload or test of live signal/thermal behavior was run.
- Remaining acceptance gaps: only the separate synthetic native harness inspects logits, repeated-token loops, and GDN/KV byte continuity; the production-agent path checks token/state telemetry but not those values. Per-turn VRAM still uses before/after samples; the guard's 2 s samples remain in a separate file. No full 64K run or performance A/B exists, and airflow remains unconfirmed.

## 2026-10-12 03:51 tokenizer target launch-path repair

- Fast-forwarded Z840's clean Lake worktree to `1550ec2`. The guard syntax/self-check passed, but the new agent-mode preflight exposed that its prior tokenizer-counter path `build/m31-agent64k-host/...` is absent there; it would refuse to launch before touching the GPU.
- Unified production server and guard to use `build/m31-accel-0001/miinfer-m31-agent-token-count`, the same configured build tree as the server. Machinist fast-forwarded to `1550ec2`, built that target in the pinned no-device ROCm container, and confirmed it executable with SHA-256 `900f5dfafc886d7f5e9f2d6389634bda92aa1e9cac0f33c6db5b1d3316b10614`. Build log: `/home/machinist/Development/MIInfer-m31-captures/token-counter-main-build.log`.
- Z840's corrected-path binary is not yet verified. No GPU ran. Next safe check: sync the follow-up commit and build/check all agent-mode executables on Z840 without device passthrough; actual inference remains gated on airflow confirmation.

## 2026-10-12 03:55 Z840 corrected-path validation

- Z840 fast-forwarded cleanly to `909698e22309510e26bcc50ea0b146f7778fc221`. In the existing pinned ROCm 7.2.1 container with `--network=none` and no GPU device nodes, `miinfer-m31-agent-token-count` built in the shared `build/m31-accel-0001` directory and is executable; SHA-256 `900f5dfafc886d7f5e9f2d6389634bda92aa1e9cac0f33c6db5b1d3316b10614`.
- The tokenizer-backed production workload self-check passed on Z840: exact initial prompt 65,536 tokens and 65,536 token IDs. Log SHA-256 `b8e9f5e3c31b9868479408d706dd1abdd2fdd9f3f65bf79e9dec4e7b1a00c2c2`. Guard syntax and threshold self-check passed. The first attempt lacked the SELinux label override and could not read the model; rerunning with the guard's existing `--security-opt=label=disable` succeeded.
- This validates the launch path and exact input construction only. No server request or GPU inference ran. The MI50 remains held because passive-heatsink airflow is unverified; 64K live state continuity and decode/prefill performance remain unqualified.

## 2026-10-12 03:58 Z840 mode-binary preflight

- Z840 synced to pushed `f2590dc02543808e902f6927344d18cb6e0d5d6e`; the Lake worktree is clean. `bash -n` passed for both launch scripts and `run-guarded.sh --self-check` passed.
- All mode-required executables are present and executable: server `6b8beebca9fcd2c08e10d2d890c9a9796c945e5813de99f22329f0e51378ba21`; tokenizer `900f5dfafc886d7f5e9f2d6389634bda92aa1e9cac0f33c6db5b1d3316b10614`; native 64K harness `651a307f54e2761afdf3438dbce13605f6e97dbb1cc22566dc12d2244a8453c6`; prefix benchmark `132a36d5bf8902b05c6b6cdba9d3aed9ce1759f46c1c16215dd6a2d21d0c1eda`; decode benchmark `6e49f10b3e6a1865bd9926211ff4226058af57df287979524e3b1085d3638884`.
- No MIInfer/tmux run is active and the MI50 remains idle at 26 C. This establishes executable preflight only; physical airflow is still unconfirmed, so no GPU work or performance measurement ran.
## 2026-10-12 04:04 — guarded production 64K attempt stopped on thermal warning

- Z840's `agent64k-05` run used source `1ac1be27174227a93c9515716290e590f0612c9b`, pinned model SHA-256 `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`, and capture directory `/mnt/lc9-lake/m31-accel-0001-captures/agent64k-05`.
- At 04:04:41 Europe/Amsterdam, junction reached the guard's 85 C inspection threshold. Guard log: `STOP junction=85.0C warning_threshold=85C; operator inspection and a fresh capture are required`; exit status 122. The stopped container is absent and the tmux pane is dead. The guard behaved as intended.
- Server startup succeeded and requests 1–2 were received; request 3 queued. No response was emitted and runner.log is empty: 0/8 turns and 0 requests completed. The persisted workload records an exact 65,536-token prompt and request/token-ID files. Request SHA-256 `54c6575412fb6a1d15431603fa9886488c3f55b0c2453b4327b923b24c6af1d3`; token-ID SHA-256 `e95d256169a1882a8708d7c511d271d03edc4973e912111284844be4a533bf5d`.
- At stop, sampled VRAM was 31,057,162,240 / 34,342,961,152 B and socket power 206 W. `telemetry.log` SHA-256 `276c80acc359c906814da825ff69c57758c293ff1830d2d9ed73ca7efd0073a1`; `agent-workload.json` SHA-256 `25164ddb5b44578d3c4d47640375a59f81b3ac8630a235cf731af826ca7d6251`.
- Recheck at 04:42 found 26 C junction, 0% GPU use, and fan 0 RPM. This does not verify airflow through the passive heatsink. Require physical cooling/airflow confirmation before a new GPU run. Correctness and all performance measurements remain unqualified.

## 2026-10-12 04:49 — host regressions, benchmark prep, LC-0009 handoff

- Read-only analysis of `agent64k-05` explains server request IDs 1/2/3: readiness `/readyz`, pre-request metrics GET, and the single intended chat POST. The POST remained in progress until the guard stopped it at 85 C. No retry or request duplication bug was demonstrated; no code change was made.
- Z840 software reports MI50 `fan1_input=0` and HP hwmon `pwm1_enable=2`, with no chassis fan RPM inputs; `ipmitool` is unavailable. Junction is 26 C, GPU use is 0%. This does not establish actual chassis airflow through the passive heatsink.
- In the pinned ROCm 7.2.1 container with `--network=none` and no `/dev/kfd` or `/dev/dri`, built the 12 host-only test executables from runtime source `1ac1be27174227a93c9515716290e590f0612c9b`; CTest passed 13/13 in 1.53 s. Build log SHA-256 `cf9134c38f95aa5a3908efef6e9ca7845d71c006cbd05164a362bf42b852892e`; CTest log SHA-256 `128683ab0ce5de0b3d8a36737267596f7f5480cae3fd5a39d77a8a8998aa491b`. The earlier CTest attempt found the host-test targets unbuilt; the explicit target build resolved that.
- Decode and prefix benchmark binaries passed their `--self-check` commands in the same no-device pinned image. Decode binary SHA-256 `6e49f10b3e6a1865bd9926211ff4226058af57df287979524e3b1085d3638884`; prefix binary SHA-256 `132a36d5bf8902b05c6b6cdba9d3aed9ce1759f46c1c16215dd6a2d21d0c1eda`; retained log SHA-256 `0b61087d07dbc3b5fc6412d73e5e946fb04ae7bae41def89df410634be9c38c0`.
- On a clean isolated Machinist worktree at `c00136d`, compiled the LC-0009 QSUM contract checker with host `g++` and ran `--self-check`: exit 0, `sum_mode_self_check=PASS`, no GPU use. Binary SHA-256 `04a8da582cb50978c850ee3e2dab4032b272b1fbe8c6c846695bf2f7abd7c08c`; checker evidence pushed in commit `524bece78f3e07da7ef9e5760f8ca03051dbb0f6`. This validates the ordinary-versus-QSUM sum-mode self-check only; the raw one-token decode QSUM reference remains missing, so LC-0009 stays OPEN/BLOCKED.
- Next: obtain physical cooling/airflow confirmation, then take a fresh guarded 64K capture. No existing capture can be resumed. 64K state correctness and decode/prefill A/B remain unqualified.



## 2026-10-12 05:25 — host work and stale-run reconciliation

- Z840 is reachable; Lake is mounted read/write with 1.6 TB free. MI50 telemetry reads 26 C junction and 0% use. The campaign tmux pane is terminal (`agent64k-05`, exit 122). Podman container `m31-accel-0001-agent64k-01` is only `created` with PID 0; it is not a live job. Existing `agent64k-05` capture records exit 122, 0 completed responses and 0/8 turns. No capture was resumed or overwritten.
- The clean Z840 runtime worktree `/mnt/lc9-lake/m31-accel-0001-cursor-build` remains at `1ac1be27174227a93c9515716290e590f0612c9b`; both GQA A/B worktrees are clean at their documented SHAs. Primary Z840 `m30/agent-workload-baseline` and Machinist `m30/exact-prefix-kv-reuse` checkouts contain uncommitted M30 changes and artifacts; preserved untouched.
- Machinist has no active Podman workload, named MIInfer process, or tmux server. Its primary branch has no upstream configured; do not infer campaign source identity from that dirty M30 checkout.
- Coordinator docs/evidence commit `619fc06b7f1df5b11df70ae5bca4cdd6dac7c57a` is pushed and the worktree is clean. Physical airflow through Z840's passive MI50 heatsink is still unconfirmed, so no GPU work was started. Next: after airflow is confirmed, make a fresh guarded 64K run; otherwise continue with isolated CPU-side review/build preparation and preserve all dirty M30 work.

## 2026-10-12 05:29 — non-finite logits fail closed

- Integrated reviewed candidate `088ff451dca3d3372ef6d8688780f0c001cd3bb1` into the coordinator and pushed it on the independent branch `experiment/m31-accel-0001-agent-a-acceptance`. The shared finite-logit predicate is used by the production sampler and native 64K harness; NaN and either infinity now raise a clear runtime error rather than allowing greedy selection to silently fall through. Validation is inside existing logits loops (no extra vocabulary pass), and finite-logit token ordering is unchanged.
- Host checks passed: warning-enabled C++ syntax-only compilation of both changed translation units against local ROCm 6.2 headers; an executable self-check using the exact shared helper accepted finite input and rejected NaN/+Inf/-Inf; `git diff --check` passed. A full CMake build was attempted but local ROCm 6.2's linker lacks `libxml2.so.2`; it did not reach compilation/link. No GPU was used.
- The HTTP production workload still cannot detect repeated-token loops because it receives streamed text rather than raw generated IDs. The native harness's four-token repeat detector covers only its synthetic diagnostic path. The production workload's token-count, failure, exact-prefix and recurrent-position gates remain intact, but 64K state continuity still requires a live run.
- Z840 passive-heatsink airflow is unconfirmed. No GPU inference or A/B run was started; prepared eager GQA and cold/suffix captures remain next after airflow confirmation.

## 2026-10-12 05:30 — Z840 access restored

- SSH preflight succeeded. Lake is mounted read/write with 1.6 TB free. MI50 reports 26 C junction, 0% use, and 10.9 MB VRAM; `podman ps` shows no active container.
- The retained `m31-accel-0001-agent64k-05` tmux pane is dead with exit 122 (04:04:41); the remaining tmux wrapper is stale. No inference was resumed or started.
- Physical airflow through the passive heatsink remains unconfirmed, so guarded GPU work stays held. Next: after confirmation, start a fresh 64K production capture, then run the prepared matched eager GQA A/B and cold/suffix comparison.

## 2026-10-12 05:38 — loop telemetry integrated; pinned no-device build passed

- Reviewed and integrated candidate `19fc0963e39fced471f946d081dcaa0768e7fc45` as coordinator commit `df4e1d2`. A shared four-identical-token detector now drives the native harness and private `miinfer_request` telemetry; the production runner fails closed if the signal is true or absent. No token IDs are exposed and generation defaults are unchanged. Candidate host-only CTest passed 1/1; Python telemetry self-check and `git diff --check` passed.
- Z840 clean worktree `/mnt/lc9-lake/m31-accel-0001-integrated` is at `df4e1d2`. In pinned image ID `50ff28a28cc13a3ca7333cd0c70e322d16245967c5fdf0d30a1bf57bda80a512`, with no network and no GPU device nodes, built `miinfer`, tokenizer, native 64K harness, decode benchmark, and prefix benchmark. Native, decode, and prefix `--self-check` commands passed. The tokenizer-backed runner self-check verified exactly 65,536 initial prompt tokens and 65,536 token IDs. Build log SHA-256 `d556b7b9d00482e5eb34ba81afa0a1f12a6f0b1da587e25a10715afe354d4389`; runner self-check log SHA-256 `b8e9f5e3c31b9868479408d706dd1abdd2fdd9f3f65bf79e9dec4e7b1a00c2c2`.
- Binary SHA-256: server `b7a7930318e2fc309d409fd98f1193849b419c8b48c8391317034ce022fea485`; tokenizer `900f5dfafc886d7f5e9f2d6389634bda92aa1e9cac0f33c6db5b1d3316b10614`; native harness `0a12e9ca49800c6368720106a039594110737fa9578b50faf83a1663f4c1f175`; decode benchmark `9f2d874a31893703e866bcb646e8f12e3e0a9670069a97d5d346a7907ea365f5`; prefix benchmark `8bc558b11b0515f7a1990a7958e54a3bd99bf6bd2a78fe83675b935d7b04ea29`.
- A source/evidence review confirms the next useful performance run is the five-round eager GQA A/B at P=65,536 on the retained exact request, control `16662fcbc22c478e897705a3eb6c59ee390fd88c`, candidate `527baba9f1f3915a6dc2507d1b8973089fb54827`. A production suffix comparison is not yet runnable because the stopped agent attempt retained only the first request token artifact; a successful turn is needed to capture a consecutive request pair. These runs remain behind physical airflow confirmation.
- Post-build telemetry remains 26 C junction, 0% GPU use and 10.9 MB VRAM. No inference or performance run was started.
- Staged the five same-hash binaries from the successful build into the launcher-required `build/m31-accel-0001/` paths on the clean Lake worktree. `bash -n` passed for both launch scripts and `run-guarded.sh --self-check` passed. This is ready-to-launch evidence only; the GPU workload was not started.

## 2026-10-12 06:15 — SSH recovery, dual-host recheck, benchmark prep

- Z840 SSH is available. Lake remains mounted read/write with 1.6 TB free; `/home` has 11 GB free. The unique MI50 is idle at 26 C junction with 10.9 MB VRAM reported. Sensor telemetry reports `fan1: 0 RPM`; it does not establish physical airflow through the passive heatsink, so GPU execution remains held.
- The earlier production pane is terminal with exit 122. Podman has no running containers; `m31-accel-0001-agent64k-01` remains only `Created` with PID 0 and was not resumed. The clean Lake integrated worktree is at `0477436f13c443fed89a11a1131e0e7d799049e9`, one documentation-only commit behind pushed coordinator `d48e416`. Guard syntax and self-check pass, and the five staged binary hashes still match the recorded build.
- Z840's actual primary checkout is `/home/fedora-workstation/Development/mi50`, branch `m30/agent-workload-baseline` at `dc83f07`; it has four modified tracked files and three untracked artifact paths and was left untouched.
- Machinist's 06:14 recheck found no active workloads, 120 GiB available RAM and 761 GB free disk. Its primary M30 checkout remains dirty, and its coordinator checkout has an unexplained local edit to `bench/prefix_state_reuse_bench.cpp`; neither was modified or used.
- Both clean GQA worktrees already have their decode binaries at both documented and guard-required paths, with identical per-revision hashes (control `ed631a655d2c4c802de026ef0cdde705b7ee583ed8334fefa84d956fe2df3c08`; candidate `026ff260630d4a7e63a3ea4d02ead847699d4cf31191d4d3d6805ef2ce7097a9`). The retained production token artifact exists (622,096 bytes).
- Updated the eager GQA recipe to alternate first revision by round, use a fresh capture root, and validate five samples per expected source, source/binary identity, successful guard exit, common model/input hashes, and generated-token parity before computing medians. The extracted shell recipe passes `bash -n`; its jq gates and metadata checks pass against a ten-capture synthetic fixture. No GPU correctness or performance result was obtained. A fresh guarded 64K run remains the next GPU action after physical airflow is confirmed; decode/prefill A/B and LC-0009 QSUM decode remain open.

## 2026-10-12 06:19 — no-device GQA binary preflight

- Ran both staged control and candidate binaries' `--self-check` in the pinned ROCm container with `--network=none`, read-only source mounts, and verified absence of `/dev/kfd` and `/dev/dri`; both printed `PASS` and exited zero. The first check attempt omitted the guard's `--security-opt=label=disable` option and exited 139 without output; repeating with the guard's label option succeeded. No GPU workload ran. This validates the preflight binaries only, not inference or performance.

## 2026-10-12 06:22 — stopped capture and state-continuity review

- Read the Z840 `agent64k-05` capture directly. It records exit 122, 0/8 turns, 0 completed requests, and retains the 65,536-token request and token-ID artifacts. Rehashed request JSON (`54c657...`) and semantic token IDs (`e95d...`); the token artifact's file hash is separate. Its 94 full ROCm snapshots span 212 seconds from 26 C to 85 C; peak sampled VRAM was 31,057,162,240 B and peak socket power 238 W. The thermal guard stop remains the demonstrated blocker.
- Independent source review at `ed8da6a` found no deterministic request-loop defect to fix before retry. The production agent checks token counts, prefix-reuse counters and GDN positions, but does not force a disk restore or verify raw KV/GDN byte parity; the native harness's synthetic snapshot/restore check is separate. Treat production restore/content continuity as unqualified. Its `urlopen(timeout=1800)` is an inactivity timeout, but this was not shown to cause the stopped run; bounded generation and guard abort remain in place.

## 2026-10-12 06:23 SSH restored; cooling gate unchanged

- Z840 SSH is reachable. Lake is mounted read/write at `/mnt/lc9-lake` with 1.6 TB free; `/home` has 11 GB free. MI50 identity remains unique ID `0x21678e17348c2f7`, gfx906; it is idle at 25 C junction, 19 W, and 10.9 MB VRAM.
- No MIInfer process is active. The prior `agent64k-05` container remains `Created` with PID 0; its stopped capture is preserved on Lake.
- Fresh sysfs inventory shows MI50 `fan1_input=0`, `pwm1_enable=1`, HP `pwm1_enable=2`, and no chassis-fan RPM input. `ipmitool` is absent. These readings do not establish airflow through the passive heatsink. No GPU workload was launched; 64K correctness and decode/prefill measurements remain unqualified until safe airflow is confirmed.

## 2026-10-12 06:28 current-source host regression

- Configured the clean Lake campaign worktree at runtime source `0477436f13c443fed89a11a1131e0e7d799049e9` with `MIINFER_BUILD_TESTS=ON` in the pinned ROCm 7.2.1 container. The container ran with `--network=none`, source mounted read-only, and no GPU devices passed. Reused the already-cached nlohmann/json source from Lake; the first attempt's absent dependency mount was corrected without changing source.
- Built the 12 host-only test targets and ran `ctest -L host-only`: 13/13 passed in 1.40 s. Retained log `/mnt/lc9-lake/m31-accel-0001-captures/gates/host-only-0477436-20261012T042700Z.log`, SHA-256 `9b84dbd74467abb7381dfcea0474bedbe36e89ebae9f5fd4321c6268d826c457`.
- The tested runtime SHA is an ancestor of current coordinator HEAD `89c58227`; commits after it are documentation-only. This is host regression evidence, not GPU, 64K workload, or performance qualification. Airflow is still unconfirmed, so GPU work remains held.

## 2026-10-12 06:36 forced disk-restore probe implementation

- Added a request-scoped `X-MIInfer-Require-Disk-Session-Restore: 1` diagnostic. It bypasses the in-memory prefix match, requires the persistent-session loader to find a matching prefix, and reports whether the prefix came from disk, its token count, and disk-load duration. The production 64K runner enables it only on the first report-completion request and requires the restored prefix to equal the initial 65,536-token prompt.
- The runner self-check accepts valid restore telemetry and rejects an absent disk-source flag, a short prefix, missing fields, NaN, and negative duration. In the pinned ROCm 7.2.1 container with `--network=none` and no GPU devices, the updated `miinfer` built and host-only CTest passed 13/13 in 1.39 seconds. Log `/mnt/lc9-lake/m31-accel-0001-captures/gates/disk-restore-no-device-build-v2.log`, SHA-256 `03be49376fb7cd5d578f50eb06a6b2da6b28ca6a8a7d8eaded8ed70d761d5697`.
- The tokenizer-backed runner self-check passed at exactly 65,536 prompt tokens and 65,536 token IDs, exercising the disk-restore telemetry checks without a server or GPU request. Log `/mnt/lc9-lake/m31-accel-0001-captures/gates/disk-restore-runner-selfcheck-v3.log`, SHA-256 `b8e9f5e3c31b9868479408d706dd1abdd2fdd9f3f65bf79e9dec4e7b1a00c2c2`; runner SHA-256 `f1c7773e70c007fa5b7dfe730d5e570045a8046aa1441fc58c7e663e485e088e`. This probe demonstrates disk-loader use and prefix length only; it does not prove restored tensor identity or numerical parity.
- The first build attempt targeted a cache directory without a generated Makefile and therefore did not build; the corrected attempt used the configured host-regression directory. Airflow remains unconfirmed, so the guarded GPU workload remains held.

## 2026-10-12 06:50 current-source rebuild and provenance finding

- Fresh Z840 inspection found no active MIInfer/Podman work, Lake mounted read/write with 1.6 TB free, and the retained `agent64k-05` tmux pane dead at exit 122. MI50 remains idle at 25 C junction, 0% use, 10.9 MB VRAM; `fan1_input=0` still cannot confirm heatsink airflow. Machinist has no active campaign jobs; its primary M30 worktree remains dirty and untouched.
- Rebuilt `miinfer`, tokenizer, native 64K harness, decode benchmark, and prefix-reuse benchmark from clean source `2b41e9b` in the pinned ROCm 7.2.1 container with networking disabled and no `/dev/kfd` or `/dev/dri`. All four no-GPU self-checks passed; the agent runner again counted exactly 65,536 prompt tokens and IDs. Build log `/mnt/lc9-lake/m31-accel-0001-captures/gates/current-2b41e9b-build.log`, SHA-256 `bafb5ae4bea9ef23d38c5ab4181f21143e16084ae56c096d165ed78bbc68fb76`; self-check log `/mnt/lc9-lake/m31-accel-0001-captures/gates/current-2b41e9b-selfchecks.log`, SHA-256 `5afd815246812b280c7f4d0c69af0d1ff5c0a8afffc2dccb41d77c09a3e66fd0`.
- Staged binary SHA-256: server `6a3e5d2b3016c7d4c988499c89a533a799760c1d9616163516b719e79573da70`; tokenizer `900f5dfafc886d7f5e9f2d6389634bda92aa1e9cac0f33c6db5b1d3316b10614`; native harness `3737348cbbf71bf343cc75ebc9a69f965a54bc947a3f35d800e1913670135bab`; decode `41fbf60b43ceb0d124a16a1e78654d4af0c24d14329aeb3348cd01d5771fe75c`; prefix `a56b5daa22cde365eaebb8853b70f1d2b04f436dc906e374ca9bd28bf8ec8eb3`.
- Review then found the generated build config embeds `MIINFER_GIT_COMMIT "UNAVAILABLE"`: the container cannot follow the source worktree's host-side `.git` pointer. These binaries pass self-checks but must not produce measurement evidence. The next no-device step is to reconfigure with the shared Git metadata mounted read-only, verify the generated commit is `2b41e9b6cf2d`, rebuild and restage. Z840 SSH timed out on two bounded attempts; no more remote polling was started. Passive-heatsink airflow independently still gates GPU work.
