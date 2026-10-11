# M31-ACCEL-0001 morning report

Deadline: 2026-10-12 10:00 Europe/Amsterdam
Status: IN PROGRESS. The exact 64K production request reached the server but was stopped at the 85 C warning before a response; persistent-agent correctness and performance remain unqualified.

## Source identity and worktrees

- Clean campaign source anchor: `dc83f07f6adaaeec75a344b0155fa049553f2f7b`, fetched from Z840's committed `m30/agent-workload-baseline` branch. It descends from `origin/main` at `3991b6bf40f810b63a100c46981e16964bb5434b`.
- Coordinator branch `experiment/m31-accel-0001-coordinator` is pushed through the current status checkpoint. Agent workload branch `experiment/m31-accel-0001-agent64k` is based on the same anchor.
- The Z840 and Machinist source checkouts both contain unrelated dirty/untracked work. They remain untouched. A separate clean Z840 worktree is at `/mnt/lc9-lake/m31-accel-0001-z840`.

## 64K persistent-agent workload

Recovered pre-campaign Z840 M30 artifact (`qualification_sha=dc83f07`) describes a 40-turn coding-agent session. It reports 40 turns, all-turn minimum-generation gate PASS, visible-answer gate FAIL, prompt/context count 66,173 by turn 32 and 82,087 by turn 40. This demonstrates a session that grew past 64K, but it did not begin with exactly 65,536 tokens and is not the required campaign qualification.

Raw server latency records in the preserved log show genuine prefix hits near the end:

| Request | Prompt tokens | Reused prefix | Suffix dispatched | Restore | Suffix prefill |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 386 | 80,730 | 79,350 | 1,380 | 1.578 ms | 27.717 s |
| 390 | 82,087 | 80,730 | 1,357 | 1.270 ms | 27.933 s |

These server records contradict a script-generated aggregate: its per-turn metrics are misjoined to server events. The original output also fails the visible-answer gate, and the exit code/per-turn client completion metadata are not retained. Treat this as diagnostic evidence only. Copies and SHA-256 hashes are preserved at `/mnt/lc9-lake/m31-accel-0001-captures/z840-m30-handoff/`; the dirty source checkout and 13 GB core dump are untouched.

An isolated worker prepared a dedicated synthetic-token diagnostic with an exact 65,536 initial context, eight bounded persistent turns, a turn-four snapshot/restore, state/position checks, and an in-process 1 GiB free-VRAM guard. Its target compiled successfully, but the guarded GPU attempt reached a sampled maximum of 87 C junction during initial prefill and was stopped before it reported a completed turn. It does not qualify persistence, agent usability, or performance.

The production-path workload was then integrated at `82f6cd5`. Its CPU-only MIInfer tokenizer target built on Z840; a host-only self-check counted the constructed production `build_chatml()` request at exactly 65,536 tokens, using seven repository files and an eight-turn tool plan. The check used the pinned Qwen3.8 GGUF and no GPU. Its retained log is `/mnt/lc9-lake/m31-accel-0001-captures/agent-host/native-self-check.log` (SHA-256 `a7ad2173803692c76242352c56171899357f323901f7bcccd8010543f4e260a4`). A preceding containerized self-check failed because that container could not read the model bind mount; running the same check directly on Z840 passed. Exact token construction is verified; a guarded production GPU request was attempted below, but live server token-count agreement remains unverified.

The clean Z840 worktree also built the `miinfer` API server target with exit 0 (binary SHA-256 `106feada3db19dca0d8b02bd9f5612e647b9175efabf3553ebc99a759d77e9a8`). A guarded startup at capacity 131,072 reached 34,254,819,328 B of 34,342,961,152 B and was stopped by the 1 GiB VRAM floor (wrapper exit 123) before the first 64K request.

The current experiment uses a bounded 81,920-token capacity, enough for the exact 65,536-token start and eight short turns while remaining within the observed MI50 memory budget. The serving whitelist was extended only for this unqualified research value. Final output is capped at 256 tokens, each planned read_file must match its specified path and line range, and there is no 128K prompt run. After that change, the clean Z840 worktree rebuilt `miinfer`, the native 64K harness, and the host tokenizer counter successfully. The updated CPU-only host self-check again counted exactly 65,536 initial prompt tokens over the eight-turn plan (log SHA-256 `a7ad2173803692c76242352c56171899357f323901f7bcccd8010543f4e260a4`). It made no GPU requests. The GPU remained idle at 28 C after the build.

The next production-path attempt loaded the model at 81,920 context, used at most 31,033,126,912 B VRAM, and remained at 30 C. Before inference, the client failed while calling `git rev-parse HEAD` inside the container: the mounted host worktree uses a host-only `.git` pointer. The launcher now injects the verified host source SHA into the container environment, preserving result provenance without requiring container access to the host Git metadata. The retained failure capture is `/mnt/lc9-lake/m31-accel-0001-captures/agent64k-03/`.

On source `f01f7e9`, the retry at 81,920 context loaded successfully and accepted the initial exact-token request. VRAM peaked at 31,056,838,656 B of 34,342,961,152 B (3,286,122,496 B sampled headroom). Junction temperature reached 85 C under 100% GPU use; sampled power was 147 W and `rocm-smi` reported fan level 25% but 0 RPM. We stopped at the warning rather than continuing toward 90 C. The container exited 137; no completion stream, tool result, or turn metric exists. Junction cooled to 38 C, GPU use returned to 0%, and VRAM returned to 10,854,400 B. Capture: `/mnt/lc9-lake/m31-accel-0001-captures/agent64k-04/`; hashes: environment `26695cdd672f19fd44c17b2853ea80431e7f5f41e03fde34bf519686d075c48d`, server log `b812ba420d866ed4190e420abb9fe861129bafde423e795fe70db39e215475a2`, telemetry `085c502d189ce485bead28086bda2d6c08067f1626db53ad868ab730761f4994`, guard `ba63280fa5b803d26ca2ebb478a821d40e3d021d22409459c8ce579eac5e19ab`. Prior `EXP-0353` records the same 0-RPM fan telemetry and that host-side fan control required sudo. Do not run another sustained GPU workload until Z840 airflow or cooling control is confirmed.

After cooldown, `sensors -u` still reported GPU `fan1_input=0`; HP WMI exposes only `pwm1_enable=2` (automatic mode), with no chassis-fan RPM through this account. `ipmitool`, `hpasmcli`, and `hponcfg` are unavailable. The retained sensor snapshot `/mnt/lc9-lake/m31-accel-0001-captures/agent64k-04/cooling-sensors-after-run.txt` has SHA-256 `dc4b7f030d964a7ef2f309dbb9810a0cc95d325146ad342fc75a43f7a6add900`. SSH telemetry cannot establish airflow through the passive MI50, so physical inspection or authorized cooling control is required before another sustained run.

Code review added a fail-closed reuse gate: each turn after the first must report both reused prefix tokens and dispatched suffix tokens on its first request. The production workload returns failure if any later turn lacks either measurement. This is an operational reuse gate, not cold-versus-reuse token-parity evidence; the known M31-0002 divergence remains unresolved.

## Decode and prefill

No campaign A/B or decode implementation yet. The existing direct/eager decode harness can run a fixed P=2048, TG=128, five-sample decode (`miinfer-v2-0045-qual-bench MODEL decode 2048 128 5`), but it lacks logit parity and does not measure the default HIP Graph route. Existing one-token attribution used a different graph-disabled route, so it does not justify a decode implementation.

The suffix worker's 64K-only prefix benchmark parity fix (`60dfa`) was cherry-picked into coordinator commit `1b477ab`. A pinned ROCm 7.2.1 no-device build from that coordinator source passed: log `/mnt/lc9-lake/m31-accel-0001-captures/build/coordinator-1b477ab-prefix-build.log`, SHA-256 `0ded1e258db670cdc76718e921c9a28e2e420b1790a3ff0157514ec95fcecf58`; binary SHA-256 `34541d9e2f37c0598ad818f62b6a1d47c29161648893efab4a5e7a5d374c5c18`. No GPU benchmark was run, so this is build evidence only, not a suffix TTFT result or correctness qualification. The older paired benchmark remains unsuitable because it combines a 128K scenario and did not fail closed on printed parity errors.

The best supported current direction is to finish 64K reliability first. For performance work, collect current matched timings and correctness before selecting a candidate. Historical eager GQA, attention tiling, and integrated prefix-reuse records are not promotion evidence.

## Memory, environment and build

- Z840 model SHA-256 verified live: `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.
- Local ROCm 7.2.1 development image base digest matches the documented pin: `sha256:fb0de294a2919ff6d503b4845e055c7fb80db5861954694ee5228e48a14411ec`.
- Z840 was idle before the campaign build: 29 C junction, 0% use, approximately 10.9 MB VRAM used. Lake is mounted read/write with 1.6 TB free.
- Initial build using the preset alone failed at CMake `find_package(hipblas)` because `CMAKE_PREFIX_PATH` was absent. The qualified-host script sets `/opt/rocm-7.2.1`; a corrected, persistent compile is underway in isolated `build/m31-accel-0001`, with output at `/mnt/lc9-lake/m31-accel-0001-captures/build/release-build.log`. No GPU device was passed to the build container.
- The guarded diagnostic sampled 30,387,605,504 B of 34,342,961,152 B total VRAM (approximately 4.0 GB free). Sampled junction temperature peaked at 87 C after the 85 C warning threshold; the run was stopped before the 90 C hard limit. Following the stop, the GPU returned to idle at 34 C with 10,854,400 B in use. The recorded telemetry is at `/mnt/lc9-lake/m31-accel-0001-captures/native64k/telemetry.log` (SHA-256 `741006b9528a19d97b5c021d3790c765c83ebf713a3c29144bdc54910b55d593`).
- The first temporary guard wrapper recorded exit 0 despite the retained container inspection showing exit 137 after manual stop, and captured no flushed workload output. The tracked replacement checks shell threshold boundaries, follows container logs and waits for final exit status. Native per-turn output now flushes immediately. The GPU remains idle; no repeat should run without active operator review of the 85 C warning trend.

## LC-0009 handoff

CPU ordinary-512 evidence was pushed at `a5140f8`; its frozen visitor-use clause failed because output omitted visitor-use facts. Full Contract O QSUM decode coverage remains open. The 24-cell Lake inventory and verified raw hashes are preserved.

## Results and disposition

No performance gain or 64K persistence pass is claimed. The recovered M30 run is diagnostic, and the synthetic-token attempt was stopped before a completed turn. The production 8-turn workload constructed an exact 65,536-token initial request, loaded at 81,920 context, and reached `request_id=3 state=queued`; it was manually stopped at the 85 C warning before a response or completed turn. Therefore live token-count agreement, tool-call correctness, multi-turn reuse, state continuity, and performance remain unverified.

## Build checkpoint

The pinned ROCm 7.2.1 container successfully compiled `miinfer-prefix-state-reuse-bench`, `miinfer-v2-0045-qual-bench`, and the updated `miinfer-m31-accel-0001-agent64k` from the clean campaign worktree. The first preset-only attempt failed because it omitted `CMAKE_PREFIX_PATH=/opt/rocm-7.2.1`; the documented path fixed it. Build exits were 0. The last build did not access the GPU. The later guarded GPU attempts are documented above; one exposed a container provenance bug, which was fixed and pushed at `f01f7e9`, and the next reached the exact 64K request before an operator thermal stop.

## Next action

Confirm Z840 airflow across the passive MI50 or provide authorized host-side cooling control, then resume the guarded 64K production request in a fresh capture. Preserve the 85 C warning and stop on unresolved cooling behavior; after a completed first turn, continue the eight-turn reuse and state gates before starting decode or prefill A/B work.

## 2026-10-12 01:22 checkpoint

Z840 SSH access is restored. Lake is mounted read/write with 1.6 TB available. The MI50 is idle at 28 C junction with 0% utilization and 10,854,400 B VRAM in use; its fan telemetry remains 15% / 0 RPM, so physical airflow is still unverified and no new GPU run was started. The production workload now persists a partial artifact before requests, reports expected/completed turns and overall workload completion, and records `replayed_tokens` from the server's `suffix_tokens_dispatched` metric. This is an artifact-quality improvement only; 64K persistence remains unqualified.

The bounded LC-0009 QSUM handoff is pushed separately at `f78949cf4f002c72a27de842ef869b6b177dc1b1`. It found no tracked raw one-token QSUM decode operand capture; LC-0009 remains open for that capture and broader Contract O evidence.

The coordinator change is pushed at `fa03664ccf021a2127f9f7110aa350ebc55e9977`, and the isolated Z840 campaign worktree is clean at that SHA. The host-only workload self-check again passed with exactly 65,536 initial ChatML tokens, seven source files, and eight planned turns; the guarded-runner self-check passed as well. Neither check used the GPU. The first invocation pointed at the wrong existing tokenizer binary path and failed before launch; correcting the path resolved it without rebuilding.

The Z840 worktree was subsequently advanced to the latest docs-only coordinator checkpoint `e6fa5592770fb7bbc78954cf7acbf1387300b735`; it remains clean.

## 2026-10-12 01:31 checkpoint

The production API workload now rejects an SSE response unless it observes both `[DONE]` and an accepted `finish_reason` (`stop`, `length`, or `tool_calls`). Its self-check covers a valid stream and a truncated stream without network access. Before every API request it atomically writes the active turn and phase to the workload JSON, then saves each completed request; a stop during inference therefore leaves a valid artifact identifying the in-flight operation. Python syntax and the no-network SSE self-check passed locally. No GPU was used. This improves failure evidence, not 64K qualification.
