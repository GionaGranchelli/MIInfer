# M31-ACCEL-0001 morning report

Deadline: 2026-10-12 10:00 Europe/Amsterdam
Status: IN PROGRESS. One guarded synthetic-token 64K prefill attempt was stopped at the thermal warning before a completed turn; the production coding-agent workload remains unqualified.

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

The production-path workload was then integrated at `82f6cd5`. Its CPU-only MIInfer tokenizer target built on Z840; a host-only self-check counted the constructed production `build_chatml()` request at exactly 65,536 tokens, using seven repository files and an eight-turn tool plan. The check used the pinned Qwen3.8 GGUF and no GPU. Its retained log is `/mnt/lc9-lake/m31-accel-0001-captures/agent-host/native-self-check.log` (SHA-256 `a7ad2173803692c76242352c56171899357f323901f7bcccd8010543f4e260a4`). A preceding containerized self-check failed because that container could not read the model bind mount; running the same check directly on Z840 passed. Exact token construction is now verified, while actual GPU execution and live server-count agreement remain unverified.

The clean Z840 worktree also built the `miinfer` API server target with exit 0 (binary SHA-256 `106feada3db19dca0d8b02bd9f5612e647b9175efabf3553ebc99a759d77e9a8`). A guarded startup at capacity 131,072 reached 34,254,819,328 B of 34,342,961,152 B and was stopped by the 1 GiB VRAM floor (wrapper exit 123) before the first 64K request.

The current experiment uses a bounded 81,920-token capacity, enough for the exact 65,536-token start and eight short turns while remaining within the observed MI50 memory budget. The serving whitelist was extended only for this unqualified research value. Final output is capped at 256 tokens, each planned read_file must match its specified path and line range, and there is no 128K prompt run. After that change, the clean Z840 worktree rebuilt `miinfer`, the native 64K harness, and the host tokenizer counter successfully. The updated CPU-only host self-check again counted exactly 65,536 initial prompt tokens over the eight-turn plan (log SHA-256 `a7ad2173803692c76242352c56171899357f323901f7bcccd8010543f4e260a4`). It made no GPU requests. The GPU remained idle at 28 C after the build.

Code review added a fail-closed reuse gate: each turn after the first must report both reused prefix tokens and dispatched suffix tokens on its first request. The production workload returns failure if any later turn lacks either measurement. This is an operational reuse gate, not cold-versus-reuse token-parity evidence; the known M31-0002 divergence remains unresolved.

## Decode and prefill

No campaign A/B or code change yet. The existing direct/eager decode harness can run a fixed P=2048, TG=128, five-sample decode (`miinfer-v2-0045-qual-bench MODEL decode 2048 128 5`), but it lacks logit parity and does not measure the default HIP Graph route. Existing one-token attribution used a different graph-disabled route, so it does not justify a decode implementation. The existing paired prefix-reuse benchmark also runs 128K in the same invocation and does not fail on printed parity errors; it will not be used for the campaign run.

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

No performance gain or 64K persistence pass is claimed. The recovered M30 run is diagnostic, and the campaign-owned synthetic 64K attempt is an **aborted thermal-feasibility diagnostic** with no completed-turn evidence. Exact 65,536-token prompt construction now passes a CPU-only check against the pinned model tokenizer; the production 8-turn GPU session has not run. The campaign remains open pending the actual serving attempt and its exact live token/state/thermal evidence.

## Build checkpoint

The pinned ROCm 7.2.1 container successfully compiled `miinfer-prefix-state-reuse-bench`, `miinfer-v2-0045-qual-bench`, and the updated `miinfer-m31-accel-0001-agent64k` from the clean campaign worktree. The first preset-only attempt failed because it omitted `CMAKE_PREFIX_PATH=/opt/rocm-7.2.1`; the documented path fixed it. Build exits were 0. The last build did not access the GPU. The later guarded GPU attempt is documented above and was stopped before a completed turn.
