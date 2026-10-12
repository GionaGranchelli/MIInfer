# M31-ACCEL-0001 morning report

Deadline: 2026-10-12 10:00 Europe/Amsterdam
Status: IN PROGRESS. The exact 64K production request reached the server but was stopped at the 85 C guard threshold before any response; persistent-agent correctness and performance remain unqualified.

## Current status — 2026-10-12 05:17 Europe/Amsterdam

- Corrected the GQA A/B harness after review found it forced HIP Graph while the candidate change is in eager decode. The route-aware harness is published on `experiment/m31-accel-0001-gqa-eager-route` (`dcd51cb1ae732f9c5dbd6da623b82a5873b90b37`) and supports explicit `--route eager|hip_graph`, route-matched sample validation, and the existing thermal guard pointed at an external clean source worktree.
- Applied the identical benchmark-only change to the prepared A/B worktrees. The approved runtime control and candidate remain unchanged. Current published heads: control `16662fcbc22c478e897705a3eb6c59ee390fd88c`; candidate `527baba9f1f3915a6dc2507d1b8973089fb54827`. Both built in the pinned ROCm 7.2.1 image with no network and no GPU device nodes, passed the route-aware `--self-check`, and embed their exact clean source SHA in generated build metadata. No inference/timing sample has run.
- Control/candidate binary SHA-256: `ed631a655d2c4c802de026ef0cdde705b7ee583ed8334fefa84d956fe2df3c08` / `026ff260630d4a7e63a3ea4d02ead847699d4cf31191d4d3d6805ef2ce7097a9`. Build log SHA-256: `129a859cdf0495cd2ce0a0feca3f4433bf1ad8e8988ec590fb079e2b5e7da472` / `707bee05ba2ced1a83f6b6f34d9c7249e1964e53ca97c8c1e28f55e2c9f14781`.
- The exact first-turn production token artifact is verified at 65,536 IDs with the expected model and token-ID hashes. Z840 remains idle at 26 C junction; no process has `/dev/kfd` open. The HP sensor module could not be loaded without sudo, and software still cannot verify airflow through the passive heatsink. GPU work remains held pending physical cooling confirmation.
- Next: after that confirmation, run the guarded 64K correctness workload, then eager-route GQA control/candidate measurements on identical prompt and settings. Compare full generated token histories before attributing timing differences. Persistent 64K correctness, decode/prefill performance, and LC-0009 QSUM decode remain unqualified.

## Current status — 2026-10-12 05:09 Europe/Amsterdam

- The exact first-request token artifact from `agent64k-05` is retained and independently verified: 65,536 IDs, expected model SHA-256 `7e78da5d…fe169`, and matching recomputed ID hash `e95d2561…3bf5d`. It is suitable as a shared fixed prompt for the next decode comparison; the stopped run has no generated follow-up history.
- A read-only audit found that the published GQA A/B comparison as currently wired cannot measure the candidate change. Commit `8a9caec` changes eager GQA decode after the HIP Graph path returns, while the benchmark always requests HIP Graph and rejects non-graph samples. The branch builds and self-checks remain valid, but no GQA performance claim can come from that setup. Harness/guard tooling is being revised for an explicit eager route before any GPU timing.
- Z840 and Machinist are reachable and idle. Z840 reports 26 C junction, 0% GPU use, and no `/dev/kfd` users. Software cannot verify chassis airflow through the passive heatsink; GPU work remains held pending physical confirmation.

## Current status — 2026-10-12 05:02 Europe/Amsterdam

- Z840 SSH is back. Lake is mounted with 1.6 TB free; the MI50 is idle at 25 C, 0% use, and 10.9 MB sampled VRAM. No GPU run started. Software still cannot establish physical airflow through the passive heatsink, so the cooling gate remains closed.
- A fresh ROCm SMI check reports 25 C edge / 26 C junction, 0% use, and 0 RPM. No process has `/dev/kfd` open. The apparent `agent64k-05` process match is a retained tmux wrapper with a dead pane (guard exit 122 at 04:04:41), not active inference.
- The prepared GQA A/B benchmark branches are now pushed and verified on `origin`: control `7ee6b3dedf16c1f3a25794ac46e3779e6f91b2cb`, candidate `7bcca14a205dab71923b3d24611db162172f3eb5`. They apply the same benchmark-only overlay to the approved direct-parent revisions `016472160c295eb1927b1ed6e095719ada6c3d18` and `8a9caec3366512e459a93bad68adde181225304b`. Both compiled in the pinned ROCm 7.2.1 image without device access and passed `--self-check`; no timing samples were collected.
- Z840 cannot authenticate to GitHub over SSH (`Permission denied (publickey)`). The two verified refs were transferred as a thin Git bundle and pushed through the coordinator's configured HTTPS remote; GitHub `ls-remote` returned the exact heads above.
- Next: after physical cooling/airflow is confirmed, run a fresh guarded 64K state/correctness capture, followed by interleaved matched control/candidate measurements. The earlier 85 C-stopped capture cannot be resumed. Persistent state correctness, decode/prefill performance, and LC-0009 QSUM decode evidence remain unqualified.

## Current status — 2026-10-12 04:42 Europe/Amsterdam

- Z840 SSH is available. The guarded production-agent attempt `agent64k-05` stopped at 04:04:41 when junction temperature reached 85 C. The guard recorded `STOP junction=85.0C warning_threshold=85C; operator inspection and a fresh capture are required` and exited 122, as designed. No container remains active; current junction temperature is 26 C, GPU use is 0%, and fan telemetry still reads 0 RPM. Physical airflow through the passive heatsink is not verified.
- The pinned model and server started; the server received requests 1 and 2 and queued request 3. The runner produced no output: 0/8 turns and 0 requests completed. The capture preserved `agent-workload.json`, the exact 65,536-token request and token-ID files, environment, server, guard, and telemetry logs. Initial request SHA-256: `54c6575412fb6a1d15431603fa9886488c3f55b0c2453b4327b923b24c6af1d3`; token-ID SHA-256: `e95d256169a1882a8708d7c511d271d03edc4973e912111284844be4a533bf5d`. Capture directory: `/mnt/lc9-lake/m31-accel-0001-captures/agent64k-05`.
- At the warning, sampled VRAM was 31,057,162,240 / 34,342,961,152 B and socket power was 206 W. The telemetry log SHA-256 is `276c80acc359c906814da825ff69c57758c293ff1830d2d9ed73ca7efd0073a1`; the workload JSON SHA-256 is `25164ddb5b44578d3c4d47640375a59f81b3ac8630a235cf731af826ca7d6251`.
- Next action: verify chassis airflow/cooling physically before any fresh GPU capture. Do not resume this capture; it is terminal. 64K state correctness, matched performance A/B, and LC-0009 QSUM decode evidence remain outstanding.

## 2026-10-12 04:49 — host regressions, benchmark prep, and LC-0009 handoff

- The captured server IDs are expected, not three generation attempts: request 1 is `/readyz`, request 2 is the runner's pre-request metrics GET, and request 3 is the single intended chat POST. The guard stopped that POST before SSE completion. The persisted `in_progress` state is therefore an interrupted request, not a retry loop. No runner/API code change is indicated by this capture.
- Z840 exposed `amdgpu` fan telemetry at 0 RPM and HP hwmon PWM mode, but no chassis fan RPM inputs; `ipmitool` is not installed. Current junction remains 26 C and GPU use 0%. Software still cannot confirm airflow through the passive heatsink.
- Rebuilt the 12 host-only test executables from capture source `1ac1be27174227a93c9515716290e590f0612c9b` in the pinned ROCm container with no network and no `/dev/kfd` or `/dev/dri`. CTest passed 13/13 host-only cases in 1.53 s. Build log SHA-256 `cf9134c38f95aa5a3908efef6e9ca7845d71c006cbd05164a362bf42b852892e`; CTest log SHA-256 `128683ab0ce5de0b3d8a36737267596f7f5480cae3fd5a39d77a8a8998aa491b`.
- Rechecked both prepared benchmark binaries in the same pinned image, with no GPU devices: decode `--self-check` and prefix `--self-check` passed. Binary SHA-256 values remain decode `6e49f10b3e6a1865bd9926211ff4226058af57df287979524e3b1085d3638884` and prefix `132a36d5bf8902b05c6b6cdba9d3aed9ce1759f46c1c16215dd6a2d21d0c1eda`; retained self-check log SHA-256 `0b61087d07dbc3b5fc6412d73e5e946fb04ae7bae41def89df410634be9c38c0`.
- Machinist completed a separate host-only LC-0009 QSUM checker build/self-check at `c00136d`; `sum_mode_self_check=PASS`, binary SHA-256 `04a8da582cb50978c850ee3e2dab4032b272b1fbe8c6c846695bf2f7abd7c08c`. Evidence is pushed at `524bece`. This does not provide the missing one-token GPU decode QSUM capture; LC-0009 remains OPEN/BLOCKED.
- The next GPU action remains a fresh capture after physical airflow/cooling is confirmed. Independently, 64K state correctness and decode/prefill performance measurements remain unqualified.

## Current status — 2026-10-12 03:08 Europe/Amsterdam

- At this 03:08 checkpoint, the coordinator branch was clean and pushed at `2961d7c25d7791339373a081307e60821279b98d`; integrated runtime code remains `a9480ae062bb83def6bb2f60e3b68a74b377bf9b`, followed by documentation-only commits.
- Z840 SSH is available. Lake is mounted read/write with 1.6 TB free. The MI50 is idle at 27 C, 0% utilization, 20 W, and 10.9 MB VRAM; fan telemetry is 15% / 0 RPM. This does not verify airflow through the passive heatsink, so GPU work remains paused after the earlier 85 C stop.
- Re-inspected `agent64k-04`: its capture contains environment, server, telemetry, and guard logs, but no saved request JSON or completed runner output. Source was `f01f7e9a210489139d7aa36e4391ede48e723464`; server requests 1–2 were received, request 3 was queued, and the container ended with exit 137 before a response. Exact request artifacts are now saved by runner `a9480ae` for a future safe attempt.
- A recursive scan also found no files in the old `session-cache` and no request/prompt JSON anywhere in the capture, so the old input cannot be recovered. Source review confirms there is no direct `decode_step()` test: recurrent views copy cursor metadata, while the fix affects the HIP convolution-history address. A host-only assertion would not validate that behavior; GPU state validation remains required.
- Completed 64K state correctness, decode/prefill A/B measurements, and LC-0009 QSUM decode evidence remain outstanding. Next: establish safe airflow/cooling, run the guarded production 64K correctness workload, then perform the prepared matched cold/suffix comparison.

## 2026-10-12 03:08 — live cooling and persistent-session path

Z840 recheck at 03:08 found no active MIInfer process or container. The MI50 is idle at 27 C junction, 0% use, 20 W; both `rocm-smi` and `sensors` still report 0 RPM. Lake remains mounted with 1.6 TB free. The airflow gate remains unresolved.

Source review verified the production snapshot path end-to-end: `run-agent-server.sh` sets session reuse and a persistent session directory; the streaming API handler enables prefix reuse and checkpoint capture; `PrefillV2Model::generate()` restores matching prefixes and persists the processed prompt. Runtime request logs expose restore time, reused GQA tokens, and GDN checkpoint/final positions. This prepares the evidence contract for the next run; no live snapshot/restore or 64K response has yet been observed.

## 2026-10-12 03:10 — Z840 cooling configuration check

Live sysfs identifies the remote host as an HP Z840. HP's service guide says the optional 1125 W configuration includes two front system fans and supports up to two 300 W GPU/compute cards; AMD lists the MI50 as passive-cooled with 300 W peak TDP. This shows the chassis family has an appropriate high-power configuration, but it does not establish which PSU/fan configuration is installed or that air is moving through this MI50 now. The live card remains idle at 27 C and its fan reads 0 RPM. Awaiting the requested physical check of chassis airflow before GPU work.

Sources: [HP Z840 maintenance and service guide](https://h10032.www1.hp.com/ctg/Manual/c04823811.pdf), [AMD MI50 specifications](https://www.amd.com/en/support/downloads/drivers.html/accelerators/instinct/instinct-mi-series/instinct-mi50-32gb.html).

## Source identity and worktrees

- Clean campaign source anchor: `dc83f07f6adaaeec75a344b0155fa049553f2f7b`, fetched from Z840's committed `m30/agent-workload-baseline` branch. It descends from `origin/main` at `3991b6bf40f810b63a100c46981e16964bb5434b`.
- Coordinator branch `experiment/m31-accel-0001-coordinator` is pushed through the current status checkpoint. Agent workload branch `experiment/m31-accel-0001-agent64k` is based on the same anchor.
- The Z840 and Machinist source checkouts both contain unrelated dirty/untracked work. They remain untouched. A separate clean Z840 worktree is at `/mnt/lc9-lake/m31-accel-0001-z840`.

## Implementation provenance

The integrated campaign implementation is coordinator code SHA `a9480ae062bb83def6bb2f60e3b68a74b377bf9b`, based on the anchor above. Relative to that anchor, the changed-file inventory is: `CMakeLists.txt`; `bench/README.md`; `bench/m31_accel_0001_agent64k.cpp`; `bench/m31_accel_0001_agent_workload.py`; `bench/m31_accel_0001_decode_bench.cpp`; `bench/m31_agent_token_count.cpp`; `bench/prefix_state_reuse_bench.cpp`; `docs/M31-ACCEL-0001-agent64k.md`; `experiments/EXP-V2-0014-prefix-state-reuse.md`; `experiments/m31-accel-0001/REPORT.md`; `experiments/m31-accel-0001/STATUS.md`; `experiments/m31-accel-0001/run-agent-server.sh`; `experiments/m31-accel-0001/run-guarded.sh`; `include/miinfer/prefill_v2/model.hpp`; `src/prefill_v2/model.cpp`; and `tools/miinfer_cli.cpp`.

Published worker branches and reviewed integration commits:

- [Agent 64K branch](https://github.com/GionaGranchelli/MIInfer/tree/experiment/m31-accel-0001-agent64k), tip `9f2900cd42f48f4d6869e2afda2239f099238a98`; its tokenizer-verified workload change was cherry-picked into coordinator `82f6cd5`.
- [Decode benchmark branch](https://github.com/GionaGranchelli/MIInfer/tree/experiment/m31-accel-0001-decode), tip `26b5f076e22ec29cdfd5609cacc7526ee2773b33`; reviewed and cherry-picked as coordinator `0ee18e6`.
- [64K suffix benchmark branch](https://github.com/GionaGranchelli/MIInfer/tree/experiment/m31-accel-0001-suffix64), tip `60dfa93554ad2124368c0db84190a60fa55e82ff`; its fail-closed parity fix was cherry-picked as coordinator `1b477ab`.
- [LC-0009 QSUM checker branch](https://github.com/GionaGranchelli/MIInfer/tree/experiment/m31-lc-0009-qsum-checker), tip `c00136d00c4e5a00c4b5460a1e31f793d4fd435e`; kept separate. Checker and ordinary-capture validation passed, but no raw one-token QSUM decode capture exists, so LC-0009 remains open.

The coordinator fixes and harnesses are not performance results: the GDN cursor changes still need GPU state validation, and no current-source decode or cold/suffix timing has been measured.

## 2026-10-12 02:34 — cursor and cold-start gate validation

Source review found two correctness gaps before another live attempt. Recurrent storage cursors could remain stale during multi-chunk prefill because they were synchronized only after the chunk loop; the cursor fix at `bf5f6d0bd991939bf23ac5af6878350c6ac9621b` synchronizes cursors around ordinary and profiled forward calls. Separately, the production runner treated turn one's prefix gate as passed without requiring a cold request. Coordinator commit `e145b71` now requires zero reused prefix/GQA/GDN checkpoint and full prompt dispatch on the first request, with failing self-check fixtures for stale-session reuse and short cold dispatch.

The clean Z840 worktree built both changes in the pinned ROCm 7.2.1 container with no GPU devices passed (exit 0). Build log SHA-256: `86272bd30fd1b3f7102a1658b9bb3ffaeeca962e7635725fd059fcac19c82e68`. The native 64K harness self-check passed for eight turns and 65,536 initial prompt tokens; prefix and decode benchmark self-checks also passed. The exact pushed production workload script from `e145b71` (SHA-256 `1ba6d46abc7f871d16779927d9c0cfe5ee0f0dd9562a549592dac53a859966fb`) passed the tokenizer-backed self-check at exactly 65,536 tokens across seven source files, with SSE and atomic artifact checks. None of these runs started a server request or used the GPU. The GDN cursor change still needs live multi-chunk numerical/state validation, and the 64K production session still has no completed run.

At 02:34, MI50 telemetry was idle at 27 C junction and 0% utilization, with fan telemetry at 15% / 0 RPM. Physical airflow remains unconfirmed, so the prior 85 C stop still blocks inference and performance measurement. After cooling is confirmed, the next run remains the guarded production 64K multi-turn correctness workload.

## 2026-10-12 02:41 — decode_step recurrent cursor fix

A call-site audit after the eager-prefill cursor fix found that public `PrefillV2Model::decode_step()` still left the storage cursor unchanged. The GDN recurrent layer reads its position from a by-value `RecurrentLayerState` view; GQA instead receives the explicit `position` argument. The function now sets the 48 stored cursors to `position` before the block loop and to `position + 1` after it. This covers repeated direct `decode_step()` callers, including the end-to-end, V2-0045, and decode-stage profiling benchmarks. The production `generate()` path already synchronized recurrent cursors separately.

The source change is pushed at `b06bffd648bfb8258dc115b7584b1842cfb03883`. Z840 rebuilt the pinned ROCm 7.2.1 targets without passing a GPU device (exit 0; build log SHA-256 `dc52e1f4b87042f885f3fa4ceea4656136dd46f84ec1ce359b8f126125a619fc`). The native 64K, prefix, and decode harness self-checks passed, as did the exact tokenizer-backed 65,536-token workload self-check (check log SHA-256 `71b47d65bbb1057d8eadc683f8b0b8b75c10c08eae83d1b6956638b1cf2aa3ea`). These runs do not execute `decode_step()` or prove numerical state continuity.

Read-only performance review found no current-source phase profile that supports a decode or prefill kernel choice. The ready next performance comparison is the existing five-pair cold 65,536-token prefill versus 65,536-token prefix reuse with a 512-token suffix; it requires parity and cooling clearance and changes no default. MI50 cooling/airflow remains unconfirmed (27 C idle, 15% / 0 RPM fan telemetry), so no GPU workload or performance measurement ran.

## 2026-10-12 02:51 — Machinist cross-host CPU build

The pre-existing dirty Machinist checkout was left unchanged. A separate clean clone was checked out at the exact pushed source `b06bffd648bfb8258dc115b7584b1842cfb03883`. In Machinist's local ROCm 7.2.1 container image (`sha256:50ff28a28cc13a3ca7333cd0c70e322d16245967c5fdf0d30a1bf57bda80a512`), the `miinfer` server and 64K harness targets built successfully without passing device nodes (exit 0). The native 64K harness self-check and guard self-check passed. Build log SHA-256: `9be32e63d2d174343848e1d04cda603c081159914348ac54b88411462635f4cc`; self-check log SHA-256: `a44a04e89101059c71c8f2a4452428d316488a894bfef023e3339d49f79b0b30`.

This is a cross-host compile check only. Machinist reported its MI50 as GPU[1] at 0% utilization and 0% VRAM; no GPU workload was launched there. This does not resolve the host-to-device qualification blocker or qualify Machinist benchmark results. Z840 remains the only planned GPU execution host, and its cooling clearance is still pending.

The same clean `b06bffd` clone also built all registered host-test targets and passed all 13/13 CTest cases labeled `host-only` in the no-device ROCm container (1.34 s total, exit 0). The retained log SHA-256 is `48b95d08352ddb7a50cc35884f737800eeb089a9d2382454bafcd834ed79a714`. These tests provide general host regression coverage; they do not exercise `decode_step()` state mutation or qualify GDN/KV GPU state continuity.

AMD identifies the MI50 as passively cooled with 300 W peak TDP ([official MI50 specifications](https://www.amd.com/en/support/downloads/drivers.html/accelerators/instinct/instinct-mi-series/instinct-mi50-32gb.html)). The board's 0 RPM sensor therefore does not establish chassis airflow. Physical airflow through the heatsink remains unverified, and the earlier 85 C stop continues to gate GPU work.

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
- Initial build using the preset alone failed at CMake `find_package(hipblas)` because `CMAKE_PREFIX_PATH` was absent. The qualified-host script sets `/opt/rocm-7.2.1`; corrected no-device builds later completed successfully, as recorded in the 02:34 and 02:41 checkpoints below. No GPU device was passed to those build containers.
- The guarded diagnostic sampled 30,387,605,504 B of 34,342,961,152 B total VRAM (approximately 4.0 GB free). Sampled junction temperature peaked at 87 C after the 85 C warning threshold; the run was stopped before the 90 C hard limit. Following the stop, the GPU returned to idle at 34 C with 10,854,400 B in use. The recorded telemetry is at `/mnt/lc9-lake/m31-accel-0001-captures/native64k/telemetry.log` (SHA-256 `741006b9528a19d97b5c021d3790c765c83ebf713a3c29144bdc54910b55d593`).
- The first temporary guard wrapper recorded exit 0 despite the retained container inspection showing exit 137 after manual stop, and captured no flushed workload output. The tracked replacement checks shell threshold boundaries, follows container logs and waits for final exit status. Native per-turn output now flushes immediately. The GPU remains idle; no repeat should run without active operator review of the 85 C warning trend.

## LC-0009 handoff

CPU ordinary-512 evidence was pushed at `a5140f8`; its frozen visitor-use clause failed because output omitted visitor-use facts. Full Contract O QSUM decode coverage remains open because no raw one-token QSUM decode operand capture is present. The 24-cell Lake inventory and verified raw hashes are preserved.

## Results and disposition

No performance gain or 64K persistence pass is claimed. The recovered M30 run is diagnostic, and the synthetic-token attempt was stopped before a completed turn. The production 8-turn workload constructed an exact 65,536-token initial request, loaded at 81,920 context, and reached `request_id=3 state=queued`; it was manually stopped at the 85 C warning before a response or completed turn. Therefore live token-count agreement, tool-call correctness, multi-turn reuse, state continuity, and performance remain unverified.

## Build checkpoint

The pinned ROCm 7.2.1 container successfully compiled `miinfer-prefix-state-reuse-bench`, `miinfer-v2-0045-qual-bench`, and the updated `miinfer-m31-accel-0001-agent64k` from the clean campaign worktree. The first preset-only attempt failed because it omitted `CMAKE_PREFIX_PATH=/opt/rocm-7.2.1`; the documented path fixed it. Build exits were 0. The last build did not access the GPU. The later guarded GPU attempts are documented above; one exposed a container provenance bug, which was fixed and pushed at `f01f7e9`, and the next reached the exact 64K request before an operator thermal stop.

## Next action

Confirm Z840 airflow across the passive MI50 or provide authorized host-side cooling control, then resume the guarded 64K production request in a fresh capture. Preserve the 85 C warning and stop on unresolved cooling behavior; after a completed first turn, continue the eight-turn reuse and state gates before starting decode or prefill A/B work.

## 2026-10-12 02:05 checkpoint

The focused suffix benchmark was updated and pushed at `a8fe7c1cb7ac1de0e16e95980fdd782faa772259`. Its 64K-only mode warms suffix graph capture, runs five alternating cold/reuse pairs, checks token-history and stop-reason parity per pair, and reports medians and observed ranges. The updated source built successfully on Z840 in the pinned ROCm 7.2.1 container without a GPU device; the benchmark `--self-check` passed. This is harness validation only, not a suffix or decode measurement.

The campaign thermal wrapper now supports `prefix64k` and launches that benchmark with retained temperature and VRAM monitoring. Workload mode is checked before hardware preflight. Shell syntax and the guard self-check passed locally and on Z840 at `7c50a7a`; this remains CPU-only validation.

The benchmark was also built at the wrapper's expected `build/m31-accel-0001/` path from `f621596f8c721ea0699f98d80514eebb80e52438`, with no GPU device passed. Build and `--self-check` exited 0 (build log SHA-256 `91a0f50e8973244eb7f95c650cbff79e8edfa12b999d449a50f6a120cac528ff`; binary SHA-256 `94b155e2b30f321c32f209e19e3a4ba16132ab146b53e8ad7694a562d36a5a7f`). It is ready to run under the guard after cooling clearance; no performance result has been measured.

Machinist independently built the same benchmark from source `00f0d8e24228bda204ed187f227035ee1b368bd7` without passing GPU devices; build and self-check passed. Its binary SHA-256 matches Z840 (`94b155e2b30f321c32f209e19e3a4ba16132ab146b53e8ad7694a562d36a5a7f`). The local Machinist image digest differs from Z840's pin, so this is a compile portability check only. Machinist's pre-existing dirty checkout was not modified, and its MI50 was not used.

After correcting the decode timing unit label, source `fea33de7eb0a2a0d1582d297c607692ee308cb66` rebuilt on Z840 in the pinned image and passed `--self-check`. The updated binary SHA-256 is `24d075003138f068f66c9a9273b4f08af51ee361b7db5aa568a827ab3e518a43`; build log SHA-256 is `31c73bf0d8237bd3b0898c598079e301fa46df848244c4a2cf2c99fb0d8cc684`. Machinist rebuilt the exact changed source file (SHA-256 `c880be8ae7cd06024e7bd5fd488f2200ac31cf906add257edbf577297f9adb94`) on clean base `00f0d8e24228bda204ed187f227035ee1b368bd7`, and produced matching log and binary hashes. Its ROCm image digest differs, so this remains a CPU-only compile check. Neither host ran inference.

At the latest pushed source `974c17343d0b456822039ef18cd2a7e711978ada`, the Z840 no-device build and native harness self-check passed for eight turns and 65,536 prompt tokens (build log SHA-256 `501b5eb5318122d5cafc063807627acc7f32c3bfe65daf8af878854f4299e6db`; binary SHA-256 `2afaac0973ffd437d6be1c14cb6c44cf98a50fcb81d8326444af4f8024d19f83`). The tokenizer-backed production workload self-check passed with exactly 65,536 initial ChatML tokens, seven source files, and SSE/atomic-artifact checks. Its first container invocation lacked SELinux label override and could not read the model mount; the read-only bind succeeded with `--security-opt=label=disable`, matching the guarded launcher's setting. Tokenizer binary SHA-256 `66ca05ebd5d5932991e3fd6f88976d800b001dc817d128c57f2b710007d96b42`; successful retry log SHA-256 `411305c5104f84aca5aff43517b13465de94fa3846a10a2b464896b94d627cbe`. Neither check made a server or GPU request.

The tmux wrappers used for ad hoc no-device builds initially interpolated `$?` before starting the detached job, so their `.exit` files are not trusted as process evidence. The retained logs independently contain the full successful link/self-check output or, for the first model-mount failure, the Python traceback; outcomes above are classified from those logs. A corrected escaped-status probe recorded exit `1` for a deliberate failing command. The tracked GPU runner retains its own container status and is not affected by this shell quoting issue.

SSH access and the read/write Lake mount are confirmed. After the build, the MI50 remained idle at 27 C junction, 0% utilization, and about 10.9 MB VRAM. Fan telemetry remains 15% / 0 RPM, so physical airflow is still unconfirmed. The 64K production workload, state continuity, decode performance, and suffix performance remain unqualified. The next action remains a guarded production 64K run after cooling is confirmed.

## 2026-10-12 01:22 checkpoint

Z840 SSH access is restored. Lake is mounted read/write with 1.6 TB available. The MI50 is idle at 28 C junction with 0% utilization and 10,854,400 B VRAM in use; its fan telemetry remains 15% / 0 RPM, so physical airflow is still unverified and no new GPU run was started. The production workload now persists a partial artifact before requests, reports expected/completed turns and overall workload completion, and records `replayed_tokens` from the server's `suffix_tokens_dispatched` metric. This is an artifact-quality improvement only; 64K persistence remains unqualified.

The bounded LC-0009 QSUM handoff is pushed separately at `f78949cf4f002c72a27de842ef869b6b177dc1b1`. It found no tracked raw one-token QSUM decode operand capture; LC-0009 remains open for that capture and broader Contract O evidence.

The coordinator change is pushed at `fa03664ccf021a2127f9f7110aa350ebc55e9977`, and the isolated Z840 campaign worktree is clean at that SHA. The host-only workload self-check again passed with exactly 65,536 initial ChatML tokens, seven source files, and eight planned turns; the guarded-runner self-check passed as well. Neither check used the GPU. The first invocation pointed at the wrong existing tokenizer binary path and failed before launch; correcting the path resolved it without rebuilding.

The Z840 worktree was subsequently advanced to the latest docs-only coordinator checkpoint `e6fa5592770fb7bbc78954cf7acbf1387300b735`; it remains clean.

## 2026-10-12 01:31 checkpoint

The production API workload now rejects an SSE response unless it observes both `[DONE]` and an accepted `finish_reason` (`stop`, `length`, or `tool_calls`). Its self-check covers a valid stream and a truncated stream without network access. Before every API request it atomically writes the active turn and phase to the workload JSON, then saves each completed request; a stop during inference therefore leaves a valid artifact identifying the in-flight operation. Python syntax and the no-network SSE self-check passed locally. No GPU was used. This improves failure evidence, not 64K qualification.

## 2026-10-12 01:33 checkpoint

Safeguards and documentation were pushed at coordinator commit `d22915b`. The clean isolated Z840 worktree fast-forwarded from `e6fa559` to `d22915b`; host `py_compile` and the full tokenizer-backed workload `--self-check` passed. Output confirmed exactly 65,536 initial ChatML tokens, seven source files, eight planned turns, SSE completion checks, and atomic artifact writing. The tokenizer self-check used no GPU. No production inference or qualification claim changed.

## 2026-10-12 01:35 checkpoint

The production-path decode benchmark from worker commit `26b5f076e22ec29cdfd5609cacc7526ee2773b33` was reviewed and cherry-picked as coordinator commit `0ee18e610f25ce0ea1f5f4a82ea018c6bcbc082f`; the coordinator branch is pushed. It adds a fixed seeded prompt, production `PrefillV2Model::generate()` HIP Graph route assertion, per-sample TTFT/decode timing, and fail-closed repeated token-history checks. It supports P=65,536 without starting a 128K workload. This is a benchmark harness, not an optimization or performance result.

Z840 SSH is live, Lake is read/write with 1.6 TB free, and the MI50 was idle at 28 C, 0% utilization, and about 10.9 MB VRAM in use. Fan telemetry still reports 15% / 0 RPM; physical airflow remains unconfirmed, so no GPU workload was launched. The exact coordinator source was checked out into a separate Lake worktree and built in the pinned `localhost/miinfer-dev:rocm-7.2.1` image (digest `bdc5ed42c985a6a333083e023225f248825a6e1511f38b6b50fbc7d5ace3fe9e`), with no GPU device passed to the container.

## 2026-10-12 01:38 build result

The pinned ROCm 7.2.1 build completed with exit 0 from source `0ee18e610f25ce0ea1f5f4a82ea018c6bcbc082f`, and `miinfer-m31-accel-0001-decode-bench --self-check` passed. Build log: `/mnt/lc9-lake/m31-accel-0001-captures/build/decode-0ee18e6-build.log`, SHA-256 `2803889d8959cee46c61c6eee8f9244ee03efd32ec915334afeae05ed06e7f31`; binary SHA-256 `900729a294a61c1a564e7007d5dbf5a381b7f164aae7bd49cd6cdb40b96431b1`. The container exited 0 and had no host devices configured. The only build warnings were existing HIP warnings about an unused parameter, loop unrolling, and occupancy. This establishes a compiled and self-checked benchmark only; there are still no decode timings. Cooling confirmation is still required before GPU execution.

At 01:40, the clean Z840 coordinator worktree was fast-forwarded from `d22915b` to pushed HEAD `c8c05c57a6902c2f99af41c4dbb5fe8a25f1feea`; it remains detached and clean. The separate build worktree at `0ee18e6` and its hashed binary/log are retained.

## 2026-10-12 02:56 — SSH recovery and runner gate correction

Z840 SSH is available again. Lake is mounted read/write with 1.6 TB free; `/home` has 11 GB free. The MI50 is idle at 27 C, 0% use, 20 W, and 10.9 MB VRAM; fan telemetry remains 15% / 0 RPM. This does not establish airflow through the passive heatsink, so the 85 C stop remains in force.

The stopped production capture `agent64k-04` contains environment, server log, guard, telemetry, and after-run cooling snapshot, but no workload JSON or captured prompt manifest/hash. Its exact run source is `f01f7e9a210489139d7aa36e4391ede48e723464`. Server log shows request 3 queued and operator stop at 85 C; the container exited 137 before a response. These missing artifacts limit reproduction and the exact prompt/token count was only host-constructed, not live-verified.

An audit found the workload runner independently tokenized only its first request and aggregated decode rate using a different token basis from the server's `decode_ms` and `decode_tokens_per_second`. Coordinator commit `0e9d612` now checks every actual request payload with the MIInfer tokenizer against per-request server telemetry, and time-weights the server decode rate by `decode_ms`. Python compilation and the no-network runner self-check passed locally; the change is pushed. Z840 tokenizer-backed validation of this newest Python revision is pending. It makes no GPU request and records no performance result.

Current integrated runner code is pushed on `experiment/m31-accel-0001-coordinator` at `a9480ae062bb83def6bb2f60e3b68a74b377bf9b`. No completed 64K production workload, matched decode/prefill measurement, or performance candidate result is established. Next: obtain physical airflow/cooling confirmation, run guarded 64K correctness first, then the prepared five-pair cold-versus-prefix suffix benchmark.

At 02:58, Z840's clean Lake worktree was advanced from `b06bffd` to the documentation-only coordinator HEAD `07a8489731421f70d768074f03e76fe062b67196`. Python compilation and the tokenizer-backed self-check passed: exactly 65,536 initial prompt tokens, eight turns, seven source files, SSE completion, and atomic artifact checks. Log SHA-256: `411305c5104f84aca5aff43517b13465de94fa3846a10a2b464896b94d627cbe`. This verifies the runner correction on Z840, without API inference or GPU access. The GPU remained idle at 27 C and 0% use, with fan telemetry at 15% / 0 RPM; airflow is still unconfirmed.

At 03:00, coordinator commit `a9480ae062bb83def6bb2f60e3b68a74b377bf9b` added an atomic sibling artifact for every exact outgoing API request body, including its SHA-256, byte count and MIInfer-tokenizer count in the workload JSON before dispatch. The artifact records model SHA from the guard and the source and runner hashes; caught request failures now persist a `failed` result with phase and active-request metadata. Local Python compilation, runner self-check (including artifact round-trip and injected failure), and guard self-check passed. Z840 tokenizer-backed `--self-check` passed at 65,536 initial tokens, 8 turns and 7 source files (log SHA-256 `411305c5104f84aca5aff43517b13465de94fa3846a10a2b464896b94d627cbe`). No API inference or GPU run occurred. This improves reproducibility of the next run but does not recover the missing input artifact from `agent64k-04` or qualify 64K correctness.

At 03:01, the separate clean Machinist clone was advanced to the exact `a9480ae062bb83def6bb2f60e3b68a74b377bf9b` source. Its no-device ROCm container built the tokenizer target and passed Python compilation plus the tokenizer-backed runner self-check with 65,536 initial tokens (log SHA-256 `05d50d96d701751a74ed75a60cb9a229f36ea7e44944de1e47f3836439566067`). No GPU device or server request was used. This independently validates workload artifact construction on the second host; it is not GPU qualification.

## 2026-10-12 02:26 — SSH recovery and LC-0009 QSUM checker

Z840 SSH is available again. Lake is mounted read/write with 1.6 TB free; `/home` has 11 GB free. No MIInfer/GPU workload or tmux session was active. The MI50 was idle at 27 C junction, 0% use, 20 W, and 10.9 MB VRAM. Fan telemetry remains 15% / 0 RPM, so physical airflow is still unconfirmed and no GPU workload was launched. The clean detached Lake source worktree is at `a1f84e3180a93f416ef72621df3a5e40cba46ac8`; the dirty `~/Development/mi50` checkout was left untouched.

The isolated LC-0009 branch [`experiment/m31-lc-0009-qsum-checker`](https://github.com/GionaGranchelli/MIInfer/tree/experiment/m31-lc-0009-qsum-checker) at `c00136d00c4e5a00c4b5460a1e31f793d4fd435e` adds a `--qsum` mode to the Contract O Gate/Up checker. It validates the QSUM field as FP16-rounded `half(d) * sum(q8_codes)` while preserving ordinary mode as the default, adds a no-model sum-mode self-check, and corrects the handoff to use the existing eager decode capture hook. The checker compiled with C++20 warnings enabled; `--self-check` passed; ordinary mode reproduced the saved Lake capture result with 160/160 Q8 groups and all 17,408 Gate and 17,408 Up rows within their bounds. The check used existing ordinary-prefill data, so QSUM decode correctness remains open until a guarded GPU capture is available.

## 2026-10-12 01:42 — per-request reuse gate

Code review found that the production agent workload only required prefix reuse on the first request of turns 2–8. A later tool-followup request could miss reuse while the turn still passed. Coordinator commit `260b9ee60355887330869d96ad80d72f3a50660b` now records a reuse gate for every request on turns 2–8 and fails the turn if any request lacks either reused-prefix tokens or dispatched suffix tokens; missing request telemetry also fails closed. Its no-network self-check exercises a passing first request followed by a failing follow-up and the no-telemetry case. The commit is pushed.

On Z840, the clean campaign worktree was advanced to `260b9ee`; Python compilation and the tokenizer-backed workload self-check passed with exactly 65,536 initial ChatML tokens, seven source files, eight planned turns, SSE completion and atomic artifact checks. Retained log `/mnt/lc9-lake/m31-accel-0001-captures/gates/request-reuse-260b9ee-selfcheck.log`, SHA-256 `64ee44bbfffe0cc57bb413a3970d353f1e97ecac59bdfaee5f86d90b36a53b25`. This used CPU tokenization only; no server or GPU inference ran. Production 64K reuse and state continuity remain unqualified.

## 2026-10-12 01:44 — deterministic suffix parity setup

The focused prefix-state reuse benchmark previously used default stochastic sampling and let Turn 1 consume RNG before the reuse run, making its cold/reuse generated-token parity check unreliable. Commit `dcde94cf49f68af1e289045bba38f09dd5bb02f5` applies identical greedy settings, neutral repetition/presence/frequency penalties, and seed 42 to cold, prefix-cache, and reuse calls; it is pushed. The pinned ROCm 7.2.1 no-device build from this source exited 0. Build log SHA-256 `9edbf83be2bd3bf8156d54a3d68f2a364c8484fc90d0907733064ad1d32c1d96`; binary SHA-256 `0f9d89f488cc5ee3d9736862796de35bd6c664aa27f3a4c542bc6e4dd419a91f`. No GPU benchmark ran. This makes the single cold/reuse parity check deterministic, but does not add repeated samples or establish reuse/state correctness.

The prefill audit found old 64K suffix attributions disagree across historical source revisions. Current campaign evidence still lacks a matched cold-versus-suffix timing. Do not select an attention optimization from the old values; first run the guarded current-source 64K correctness path, then collect matched repeated cold/suffix timings and profile on that same source.

## 2026-10-12 01:44 — exact prior-prefix request gate

The production runner's reuse gate was tightened again at coordinator commit `529cf3cd84a6c1e5969fa785797d65d23683a74c`, now pushed. Every request after the first must reuse exactly the preceding request's `prompt_tokens` and dispatch a nonempty suffix; missing prompt telemetry fails closed. The no-network self-check covers exact reuse, a one-token-short prefix, zero suffix, and missing request telemetry. On Z840, the clean campaign worktree synced to `529cf3c`; `py_compile` and the tokenizer-backed `--self-check` passed with 65,536 initial tokens, seven source files, eight turns, SSE and atomic artifact checks. Log SHA-256 `64ee44bbfffe0cc57bb413a3970d353f1e97ecac59bdfaee5f86d90b36a53b25`. No inference ran. This is an operational cache-prefix continuity gate; it does not establish numerical state parity or completed multi-turn correctness.

## 2026-10-12 01:46 — Z840 live recheck

SSH succeeded. Lake is mounted read/write with 1.6 TB free; `/` is 98% full with 11 GB free. The MI50 is idle at 28 C junction, 0% use, 10.9 MB VRAM, and 20 W. No MIInfer/build process or tmux session is active. `rocm-smi` and `sensors` report 0 RPM for the MI50 fan sensor; `ipmitool` is unavailable and no chassis-fan sensor is exposed, so software cannot confirm physical airflow. GPU work remains held after the earlier 85 C stop. The isolated suffix64 source is already integrated at `1b477ab` and its no-device build succeeded; this recheck produced no new correctness or performance result.

## 2026-10-12 01:47 — artifact revalidation

The clean Z840 campaign worktree remains at `529cf3c`. The tokenizer-backed self-check log's SHA-256 still matches `64ee44bbfffe0cc57bb413a3970d353f1e97ecac59bdfaee5f86d90b36a53b25`, and the retained guard records the production 64K attempt's operator stop at 85 C (exit 137). Idle telemetry is 28 C, 0% utilization, approximately 10.9 MB VRAM, and 20 W. No new GPU workload was run.

## 2026-10-12 01:48 — production position and capacity gates

The production runner now checks the API's `gqa_kv_reused_tokens` and `gdn_checkpoint_position` against each request's reused prefix, and requires `runtime_context_capacity >= prompt_tokens`. It records the observed values and per-request gate results. The no-network self-check passes for cold start and exact reuse, and rejects a short prefix, empty suffix, incorrect GDN position, capacity overflow, and missing request telemetry; complete/truncated SSE and atomic artifact checks also pass. Python compilation passes. This is a check of server-reported reuse positions and configured capacity, not numerical recurrent-state continuity or KV content parity. No tokenizer or GPU request was run for this change; the GPU remains held pending airflow confirmation.

On Z840 at source `4789d094932548bb70fda02ec85e0b186a6c3e9f`, Python compilation and the tokenizer-backed workload `--self-check` passed with exactly 65,536 initial ChatML tokens; the updated no-network position/capacity checks passed. Retained log `/mnt/lc9-lake/m31-accel-0001-captures/gates/position-capacity-4789d09-selfcheck.log`, SHA-256 `411305c5104f84aca5aff43517b13465de94fa3846a10a2b464896b94d627cbe`. The clean remote worktree is at that SHA. No model inference or GPU request ran.

## 2026-10-12 01:50 — GDN cursor bookkeeping candidate

Source review found cold prefill did not update the outer `RecurrentLayerStateStorage` positions: `forward()` mutates by-value views, while eager `RecurrentLayer::decode()` uses the stored position for convolution indexing. The candidate sets all 48 stored cursors to the completed prompt position, advances them after each recurrent decode position is consumed (even if the next token stops or cancels), and emits observed minimum/maximum final cursors in request telemetry. The production workload checks those reported positions against the prompt/generation bounds. Local Python compilation and the no-network self-check pass. This is a candidate correctness fix; C++ build and remote source validation are pending, and GPU/numerical behavior is unverified. The KV storage still exposes capacity but no independent valid-length/content check, so GQA cache continuity remains unqualified.

The candidate then built from `0941ae99d0f7036717538941b7ad3640082aa4b9` in the pinned ROCm 7.2.1 container without passing GPU devices. `miinfer` and `miinfer-m31-accel-0001-agent64k` both linked with exit 0. Build log SHA-256 `ef06ee966407663b0caeb7607d8d1d34112e6b6c7e8f182647c9a48ef8ccdf59`. The native harness `--self-check` passed (binary SHA-256 `53f3bff20b6e0a57b19468d8bedb0aa28dc49b7ba51f3e72c0a5de84b00f1fd1`; log SHA-256 `b51d580dbd526103240d11197dfdb2b6805567e71a65231bc84ebffd018cd9d9`). The tokenizer-backed Python workload self-check also passed from the same source with exactly 65,536 initial ChatML tokens; log SHA-256 `411305c5104f84aca5aff43517b13465de94fa3846a10a2b464896b94d627cbe`. GPU stayed idle. This verifies compilation and harness setup only; the change still needs a guarded live multi-turn check, and numerical GDN/KV continuity plus GQA cache validity remain unqualified.

## 2026-10-12 01:56 — power-cap access check

Tried to lower the Z840 MI50 cap to 100 W as a reversible load reduction while physical airflow remains unknown. Noninteractive sudo was denied because a password is required; the limit remains at its 225 W stock value. The MI50 is idle at 28 C and 0 RPM. No inference ran. A guarded 64K attempt can resume after physical airflow is confirmed or the user sets a lower cap interactively; the existing thermal stop remains in force.

## 2026-10-12 01:58 — request metric attribution gate

The production runner now fails closed unless per-request prompt-token and generated-token counters match the corresponding server request log record, with exact integer fields. A no-network self-check covers matching counts and a generated-token mismatch; Python compilation passes. This prevents aggregate decode counts from being attributed to the wrong request. Z840 validation of this latest script revision is pending; no model or GPU request ran.

On Z840, source `284435de2afd989e2677a723ce2e0df7084046d6` passed Python compilation and the tokenizer-backed workload `--self-check`: exact 65,536 initial ChatML tokens, seven source files, and updated no-network metric checks passed. Retained log `/mnt/lc9-lake/m31-accel-0001-captures/gates/request-metrics-284435d-selfcheck.log`, SHA-256 `411305c5104f84aca5aff43517b13465de94fa3846a10a2b464896b94d627cbe`. No server request or GPU inference ran.

## 2026-10-12 03:04 — Z840 SSH and capture recheck

SSH succeeded on the first attempt. Lake remains mounted read/write with 1.6 TB free. Re-enumerated `agent64k-04`; it contains no `.json` request artifact, and its server log ends after request 3 entered the queue. Recorded exit status is 137; no runner result or completed response exists. The MI50 is idle at 27 C junction, 0% GPU use, 20 W, and 10.9 MB VRAM; fan telemetry reads 15% / 0 RPM. No workload ran; physical airflow through the passive heatsink remains unconfirmed.

## 2026-10-12 03:07 — cursor validation boundary

Read-only source/call-site audit found no test directly invokes `PrefillV2Model::decode_step()`. `RecurrentLayerStateStorage::view()` copies the cursor, and recurrent decode passes it to the HIP convolution-history update; therefore a host-only cursor assertion would not exercise the changed behavior. Existing no-device build and 13/13 host-only CTests validate compilation/general host regressions only. The fix needs a guarded Z840 GPU check against fixed token history once cooling is confirmed. A recursive scan also confirms the old `agent64k-04/session-cache` and request/prompt artifact paths contain no recoverable files.

## 2026-10-12 03:13 — Z840 SSH restored

SSH succeeded twice. Lake is mounted read/write with 1.6 TB free; no MIInfer or campaign process is active. The MI50 remains idle at 27 C junction, 0% utilization, 20 W, and 10.9 MB VRAM. Current Linux hwmon exposes no chassis fan RPM; the separate `hp_wmi_sensors` module is installed but unloaded and was not loaded. This does not establish airflow through the MI50 passive heatsink, so guarded inference and performance runs remain held. The existing dirty M30 checkout was left untouched. No correctness or performance result changed.

## 2026-10-12 03:21 — reuse accounting and benchmark timing gates

Coordinator commit `38eccc295b39e8c9db35ea1ab78c13ad870c9855` tightens two fail-closed evidence gates. The production agent runner now requires integer, nonnegative dispatch counts, bounds reused tokens by prompt length, and requires reused plus dispatched suffix tokens to equal the request prompt. Its Python syntax check and direct SSE/telemetry self-check passed, including a request whose dispatched count is short by one. The 64K prefix benchmark now requires each warmup and measured cold/reuse sample to use HIP Graph and report finite, positive prefill/TTFT/decode/total timing; restore and suffix timing must be finite and nonnegative. Its no-device C++ build and `--self-check` passed on Machinist in the pinned ROCm 7.2.1 image. Binary SHA-256: `e71c1f3720a4c189568d22e1de303284d381a48c6457f024c01296b6b1e2cd98`.

The first fresh-worktree CMake configure could not resolve GitHub while fetching nlohmann/json. Reusing the dependency cache from Machinist's existing clean clone with `FETCHCONTENT_FULLY_DISCONNECTED=ON` completed the build; no device was passed to Podman. Review also confirmed the prefix and decode benchmark fixtures use synthetic token IDs, so those results must be labeled synthetic until measured with frozen tokenized coding-agent input. The MI50 remains gated on physical heatsink airflow confirmation; no live inference or performance result was produced.

At 03:22, coordinator commit `27c3ab3` added a decode-sample validity gate: each sample must use HIP Graph and report finite positive TTFT, decode duration, and decode throughput. The self-check rejects graph fallback and infinite decode time. Machinist rebuilt the target from that exact source in its cached pinned ROCm 7.2.1 image without device passthrough; the target self-check passed. Binary SHA-256: `3fa0b094cf0bb06539e66c59ff7343a32f9a4497d927e8a73d33721cf4eb095a`. This validates the measurement harness only. No A/B capture or GPU timing exists.

At 03:24, `run-guarded.sh` gained `decode` mode with P=65,536, TG=128, and five samples by default, writing the benchmark JSON into the guarded capture directory. This closes the launch-path gap so decode A/B runs can use the same thermal and VRAM sampling wrapper. Shell syntax, guard self-check, and whitespace checks passed locally. Z840 build/sync and all live measurements remain outstanding.

At 03:25, Z840 SSH succeeded and confirmed Lake mounted read/write with 1.6 TB free. The clean remote campaign worktree is at pushed `c30bcdeb8081fb0d7832d4ecc78b1df2e16fb7c3`. Its pinned ROCm 7.2.1 no-device build log SHA-256 is `942192a2218f2e9247c659e0e29ac7c2cbcfc0f5304369d8eefc12848f5e560a`; prefix and decode benchmark binary SHA-256 values are `e71c1f3720a4c189568d22e1de303284d381a48c6457f024c01296b6b1e2cd98` and `3fa0b094cf0bb06539e66c59ff7343a32f9a4497d927e8a73d33721cf4eb095a`. Prefix, decode, and guard self-checks pass. This is build and harness evidence only. The MI50 is idle at 27 C and 0% use, with 0% VRAM use and 15% / 0 RPM fan telemetry; physical airflow is unconfirmed, so no inference or performance run was made.

At pushed coordinator source `8bd8b4c6e7dae14379f921101237c8a513df61e7`, the production workload saves exact ChatML token IDs for every request next to its exact request body. Each `.tokens.json` includes the body hash, model ID, token count, and canonical little-endian token-ID hash. Z840 rebuilt the tokenizer target in the pinned ROCm 7.2.1 image without GPU passthrough; build log SHA-256 `d2f438121df46da2bd1755b044e64f2eddf33c74974975211a7e411f3fae0243`, binary SHA-256 `900f5dfafc886d7f5e9f2d6389634bda92aa1e9cac0f33c6db5b1d3316b10614`. The tokenizer-backed no-GPU self-check returned exactly 65,536 IDs for the 65,536-token initial request; log SHA-256 `b8e9f5e3c31b9868479408d706dd1abdd2fdd9f3f65bf79e9dec4e7b1a00c2c2`. This enables deriving the real common prefix for consecutive requests, but the performance benchmark binaries still use synthetic fixtures and cannot yet consume these artifacts. No live request or GPU work ran.

At pushed source `ff4ff7343822ccea1fd12b512f8393e9fc778a51`, both decode and prefix benchmark targets consume production token-ID artifacts. Decode uses one exact request; prefix calculates the token-ID LCP for two adjacent request artifacts, preloads the prior request, and measures cold versus reused execution of the next request under the existing parity, dispatch, graph, and timing gates. The shared loader validates the token artifact hash/count and vocabulary. Z840 built both targets in the pinned ROCm 7.2.1 container with no GPU device passed: build log SHA-256 `ad4487ab5c13cbd435a7e6e772b25e640b8ebcdc0068f1dd3c69b15b57d0541e`; binary SHA-256 values `5f53e5bac80c9aac82b04593de72c4ef40b00c349a2038a7ba732ff8c4d6364d` and `a5f21b0bf206d0e8e8ad952ffdca2a17e5ea7862481848d163b32aa27ce3b43e`. Both self-checks pass (combined log SHA-256 `0b61087d07dbc3b5fc6412d73e5e946fb04ae7bae41def89df410634be9c38c0`). The guarded launcher can select these artifacts from a new capture directory; its syntax and self-check pass locally.

Z840's final host-only recheck found the MI50 idle at 26 C, 0% utilization and VRAM, 20 W, and fan telemetry 15% / 0 RPM. No live inference or timing was run. Physical airflow through the passive heatsink remains unconfirmed, and no captured production token pair exists yet. The next step is still the cooling confirmation, followed by the guarded 64K agent run and then production-token decode/prefix A/B.

## 2026-10-12 03:39 — restored SSH and current-source host regression

Both Z840 and Machinist were reachable and their clean campaign worktrees were at `f2715b246256b93883176ee90e17935595f041d4`. Lake was mounted read/write with 1.6 TB free. Machinist rebuilt the registered host-test targets and passed all 13 CTest cases labeled `host-only` in 1.33 seconds. The run used the existing dependency-populated build directory inside the pinned ROCm 7.2.1 container with no GPU device nodes passed. Log SHA-256: `8e096862ea5c53d758177b8c08cf63113fac65bcb389293a028def1a851ca20c`.

A new offline CMake build directory attempted to fetch `nlohmann/json` and failed because DNS/network was unavailable. The existing configured build directory already held that dependency, so reusing it completed the regression run; no source files changed on Machinist. The reviewer rechecked `f2715b2` and confirmed the earlier artifact provenance and exact-prefix findings are resolved.

The host tests do not establish MI50 numerical state continuity or qualify inference performance. No live 64K run or performance A/B was made. Z840's MI50 remained idle; physical airflow through its passive heatsink remains unconfirmed and still gates GPU work. After cooling is confirmed, the next run is the guarded 64K multi-turn correctness capture, then token-matched decode/prefix A/B.

## 2026-10-12 03:48 — guard hardening after acceptance review

An acceptance review found two safety gaps in the guarded launcher: its detached container could outlive the SSH-side monitor, and a sample at the 85 C warning threshold only logged while continuing. `run-guarded.sh` now stops the container at 85 C pending inspection, at 90 C, on the VRAM limit, or when the monitor receives HUP/INT/TERM. The launch self-check verifies the continue/inspect/stop thresholds. Before start it also rejects a dirty source worktree, checks the executable(s) required by the selected mode, and records their SHA-256 hashes. The documented production command starts the foreground monitor in tmux and records the pane PID.

`bash -n`, the guard self-check, and `git diff --check` pass. No GPU workload or live signal/temperature test was run. The separate native synthetic harness checks finite logits, repeated-token loops, recurrent positions, and GDN/KV byte identity across save/restore. The production-agent path still does not inspect raw generated token IDs/logits or state bytes, and its per-turn VRAM maximum is boundary-sampled; 2 s guard telemetry remains separate. These are qualification limits, not passed production checks. Full 64K inference and decode/prefill A/B remain unmeasured because Z840 airflow is unconfirmed.

## 2026-10-12 03:51 — agent tokenizer build-path correction

Fast-forwarding Z840 to `1550ec2` and checking the new guard found that the expected `build/m31-agent64k-host/miinfer-m31-agent-token-count` did not exist, which would have failed agent-mode preflight. The production server and guard now use the shared `build/m31-accel-0001` tree. Machinist built the tokenizer-counter target there in the pinned no-device container; the executable SHA-256 is `900f5dfafc886d7f5e9f2d6389634bda92aa1e9cac0f33c6db5b1d3316b10614`. The docs now include the matching build command. Z840's corrected-path artifact still needs a no-device check. No GPU run occurred; airflow remains unverified.

Z840 then fast-forwarded to `909698e22309510e26bcc50ea0b146f7778fc221`. The tokenizer target built in the shared production build directory in the pinned ROCm 7.2.1 container with no network and no device nodes; binary SHA-256 matched Machinist at `900f5dfafc886d7f5e9f2d6389634bda92aa1e9cac0f33c6db5b1d3316b10614`. The tokenizer-backed workload self-check passed with exactly 65,536 initial prompt tokens and IDs. Retained log SHA-256: `b8e9f5e3c31b9868479408d706dd1abdd2fdd9f3f65bf79e9dec4e7b1a00c2c2`. The first attempt omitted the SELinux label override and could not open the model; rerunning with `--security-opt=label=disable` passed. No API request, inference, or GPU timing ran. Physical airflow through the passive MI50 heatsink remains unverified.

Z840 then synced to `f2590dc02543808e902f6927344d18cb6e0d5d6e`. Its clean worktree passes both launcher syntax checks and the guard self-check. All five mode-required executables are present: server SHA-256 `6b8beebca9fcd2c08e10d2d890c9a9796c945e5813de99f22329f0e51378ba21`, tokenizer `900f5dfafc886d7f5e9f2d6389634bda92aa1e9cac0f33c6db5b1d3316b10614`, native harness `651a307f54e2761afdf3438dbce13605f6e97dbb1cc22566dc12d2244a8453c6`, prefix benchmark `132a36d5bf8902b05c6b6cdba9d3aed9ce1759f46c1c16215dd6a2d21d0c1eda`, and decode benchmark `6e49f10b3e6a1865bd9926211ff4226058af57df287979524e3b1085d3638884`. No GPU workload ran; airflow remains unverified and gates inference.
