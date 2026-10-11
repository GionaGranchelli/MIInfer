# M31-ACCEL-0001 morning report

Deadline: 2026-10-12 10:00 Europe/Amsterdam
Status: IN PROGRESS. One historical-on-base multi-turn GPU artifact was recovered and preserved; no campaign-owned GPU run or optimization has been completed yet.

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

An isolated worker is preparing a dedicated 64K-only, eight-turn workload with inline server/model metrics and no request-ID join. It must use the exact 65,536 initial token count, tool-like append-only turns, persistent prefix reuse, bounded generation, state/position checks, and explicit failure gates before any run can qualify.

## Decode and prefill

No campaign A/B or code change yet. The existing direct/eager decode harness can run a fixed P=2048, TG=128, five-sample decode (`miinfer-v2-0045-qual-bench MODEL decode 2048 128 5`), but it lacks logit parity and does not measure the default HIP Graph route. Existing one-token attribution used a different graph-disabled route, so it does not justify a decode implementation. The existing paired prefix-reuse benchmark also runs 128K in the same invocation and does not fail on printed parity errors; it will not be used for the campaign run.

The best supported current direction is to finish 64K reliability first. For performance work, collect current matched timings and correctness before selecting a candidate. Historical eager GQA, attention tiling, and integrated prefix-reuse records are not promotion evidence.

## Memory, environment and build

- Z840 model SHA-256 verified live: `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.
- Local ROCm 7.2.1 development image base digest matches the documented pin: `sha256:fb0de294a2919ff6d503b4845e055c7fb80db5861954694ee5228e48a14411ec`.
- Z840 was idle before the campaign build: 29 C junction, 0% use, approximately 10.9 MB VRAM used. Lake is mounted read/write with 1.6 TB free.
- Initial build using the preset alone failed at CMake `find_package(hipblas)` because `CMAKE_PREFIX_PATH` was absent. The qualified-host script sets `/opt/rocm-7.2.1`; a corrected, persistent compile is underway in isolated `build/m31-accel-0001`, with output at `/mnt/lc9-lake/m31-accel-0001-captures/build/release-build.log`. No GPU device was passed to the build container.
- No campaign peak-VRAM or thermal run measurement is available yet. A historical separate 64K call reported a 102 C hotspot; active telemetry/thermal guards are required for any new GPU run.

## LC-0009 handoff

CPU ordinary-512 evidence was pushed at `a5140f8`; its frozen visitor-use clause failed because output omitted visitor-use facts. Full Contract O QSUM decode coverage remains open. The 24-cell Lake inventory and verified raw hashes are preserved.

## Results and disposition

No performance gain is claimed. The recovered M30 run is diagnostic, not a qualifying 64K-start workload. The campaign remains open pending code review/build and one serialized, guarded 64K persistent-agent execution. Update this report with exact run command, output checksums, source/model/image identity, state/parity outcome, memory/thermal measurements and an explicit disposition.

## Build checkpoint

The pinned ROCm 7.2.1 container successfully compiled both `miinfer-prefix-state-reuse-bench` and `miinfer-v2-0045-qual-bench` from runtime source `dc83f07` using the documented `CMAKE_PREFIX_PATH=/opt/rocm-7.2.1` configuration. The first preset-only attempt failed because it omitted that path; no source change was needed. Build exit was 0. Binary hashes and full log are in `STATUS.md` and Lake's campaign capture directory. The GPU stayed idle during the build.
