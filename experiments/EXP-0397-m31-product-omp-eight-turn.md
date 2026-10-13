# EXP-0397 — Oh My Pi persistent eight-turn task on MI50

**Status:** PARTIAL — explicit XML tool dispatch works, and eight consecutive short warm follow-up calls have a 14.46 s median first-action time (five of eight were at or below 15 s). The full eight-turn coding task, reliable coding-action latency, and sufficient context headroom are not yet proven; do not mark the product gate passed.
**Milestone:** M31-PRODUCT-0001  
**Host:** Machinist X99 only; Z840 excluded.  
**GPU:** MI50 GPU0, `0000:04:00.0`, UUID `0x21678e17348c2f7`; GPU2 idle.  
**Model:** Qwen3.8-27B Q4_K_M, SHA-256 `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.  
**Runtime:** ROCm 7.2.1 image `sha256:447907c697fb2c39ce9657401e354a70753d5672ad955d4ceb12de976b91224a`; binary SHA-256 `1dcfd59d73ad4949e26135b445c11d5dd3a9aff05ffac0fa625a8235c0e28c84`.  
**Context:** 98,304-token capacity; MIInfer request 7 processed a 74,992-token prompt from the 246,640-character relevant-context bundle (SHA-256 `2ed2d7b915f45a1951909a5c6664fb09f2e0813bbbad07ab1d5e4dcc80350478`).

## Workload and result

Oh My Pi v18.10.2 ran in JSONL RPC mode in one session (`01a132e2-b7cc-744c-b0ec-6bd423af3459`). The eight user prompts stayed in that session. Turn 1 processed 74,992 tokens, then performed seven bash tool calls with seven tool results. The initial turn took 2,132.78 seconds end to end. It saved a growing-prefix checkpoint at 84,654 tokens (2.95 GB).

Turns 2 and 3 hit the model's 2,048-token generation limit while reasoning, with no tool calls (187.82 and 194.64 seconds). I raised the local OMP profile limit to 4,096 as a direct integration fix. Turn 4 then settled in 359.19 seconds with six tool calls and six tool results. It edited `docs/interactive-serving.md` to add an Oh My Pi endpoint example. The task required corrections because the model initially emitted invalid provider syntax; the final block now has a parseable `providers.miinfer-m31.models` structure.

After turn 4, I restarted MIInfer while keeping the OMP RPC session alive. The first follow-up request restored a persistent checkpoint: 87,092 prompt tokens, 86,688 reused tokens, `persistent_session_restored=true`, and 2,884.88 ms restore time. This proves a near-turn-four disk checkpoint can be restored and continued. A later request in the same run fell back to an older 84,366-token prefix and had to prefill 3,655 new tokens, so repeatable continuity through the changing agent transcript is not established.

## Measurements

| Path | Prompt | Reused | Restore | Prefill | TTFT wall | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cold first turn (request 7) | 74,992 | 0 | 0 | 1,431,600 ms | — | OMP turn: 2,132.78 s |
| First request after turn-four server restart (new request 2) | 87,092 | 86,688 | 2,884.88 ms | 34,873.3 ms | 37,970.5 ms | 64,210.3 ms |
| Later request that restored the older prefix (new request 5) | 88,021 | 84,366 | 2,815.36 ms | 151,473 ms | 154,533.9 ms | 159,089 ms |

The OMP transcript gives the direct user-message-to-tool-result measurement:

| Turn | First successful action | Time |
| --- | --- | ---: |
| 5, first turn after MIInfer restart | Read result | 64.376 s |
| 6 | Read result | 159.270 s |
| 7 | Successful bash replacement, after one failed call | 246.878 s |

Each is one distinct turn, not a repeated sample of an identical task; the observed span is 64.376–246.878 seconds, with no median claim. The 15-second gate fails on all three. Request-level MIInfer TTFT in the measurement table is a diagnostic metric, not a substitute for this user-visible action time. The dominant costs are suffix prefill (34.9 seconds on the first post-restart request; 151.5 seconds on the later stale-prefix restore) plus model generation/tool-call handling. No competitive engine claim is made.

GPU0 reached 75°C junction during the recorded workload; the warning/stop thresholds were 80/90°C and neither triggered. GPU2 remained idle. The user-set 180 W cap was in place.

## Correctness and integration

- The live OMP transcript records actual bash/read/edit calls and tool-result messages. It includes a tool-call parser fix for OMP's plural `<tool_calls>` wrapper; the rebuilt server passed a separate short-context tool round trip before this run.
- The turn-four checkpoint file `sess_d0b9d3d46517f75e_p86881.miinfer` was present on the mounted session volume. The first post-restart prompt restored a checkpoint at 86,688 tokens and prefetched its 404-token suffix. The exact final-turn state identity was not matched byte-for-byte; later selection fell back to 84,366 tokens.
- The Oh My Pi YAML block parses locally as the expected provider, URL, API, auth string, model id, context size, and output limit. `auth: none` parses as the string `"none"`; the final agent validation incorrectly expected YAML null, then began changing the example to `auth: null`. I stopped the retry loop, restored `auth: none`, and reran the corrected parser assertions successfully. `git diff --check` is clean.
- The coding task was a focused documentation integration change, not an application-code change. The eight-turn workload did not finish within responsiveness limits.

## Decision and next step

**Do not mark M31-PRODUCT-0001 passed.** The cold 64K+ run, real tool execution, a turn-four checkpoint, and one disk restore are demonstrated. The latency gate fails by at least 2.5× on the first post-restart response and the later stale-prefix fallback shows restoration is not consistently selecting the newest state.

Code inspection found that MIInfer tries an exact in-memory prefix first, then scans disk checkpoints and selects the longest checkpoint whose token IDs exactly match the new prompt. I compared the on-disk token arrays: `p84366` is an exact prefix of `p86688`, and `p86688` is an exact prefix of `p86881`. The later request's best disk match was `p84366`, even though `p86688` existed. The checkpoint files themselves are nested and intact.

The source also shows a likely cause: [`build_chatml()`](../src/openai_api.cpp) appends an open `<|im_start|>assistant` header to every input prompt, and [`PrefillV2Model::generate()`](../src/prefill_v2/model.cpp) saves the full prefill prompt. When Oh My Pi sends the next turn, that unfinished assistant header is replaced by the generated assistant response and tool-result messages. The old full-prompt checkpoint is no longer an exact prefix, so [`find_matching_session()`](../src/prefill_v2/persistent_session.cpp) falls back to the earlier checkpoint before that changing tail. This matches the observed drop from 86,688 to 84,366 reused tokens. A stable pre-generation checkpoint boundary or saved post-decode state is needed for efficient repeated tool turns; neither was tested in this run.

Next, prove the prompt-boundary explanation with a small tokenizer-level check, then assess the smallest safe way to checkpoint an append-stable boundary for OMP tool turns. Rerun only the turn-four restart/follow-up boundary after that correction; keep the 15-second target unchanged. Do not expand model, GPU, or kernel scope.

## 2026-10-13 candidate boundary retest

The candidate binary was built from pushed source commit `95516b722493b5bada1f3d4eb726c8e4b1f4dd9b` plus temporary diagnostic logging in `qwen3_primitives.hip` (binary SHA-256 `5a0d7c82b9704490f4b9db21989875a8831250fb73589c8c5da19746a06014d2`). It ran in the existing ROCm 7.2.1 candidate container on Machinist GPU0; GPU2 remained idle. This is a diagnostic candidate build, not the final production binary.

Oh My Pi v18.10.2 submitted the 78,245-token request through MIInfer. It executed real `bash` and `grep` tool calls against the clean `m31-omp-eight-turn` worktree and received both tool results. MIInfer completed the cold prefill in 2,908,130 ms (26.91 tokens/s), then generated 250 tokens and returned `finish_reason=tool_calls`. The request wrote persistent checkpoint `sess_604982ec315821e4_p78242.miinfer` (about 2.74 GB), exactly three tokens before the open assistant header. This confirms the requested stable boundary was captured while processing the prompt. GPU0 reached 52°C junction; no fault or thermal stop occurred.

Subsequent same-process requests reused the checkpoint boundary and did not replay the full prefix:

| Request | Prompt | Reused | New suffix | Restore | TTFT wall | Total | Finish |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 3 | 82,737 | 78,242 | 4,495 | 1.19 ms | 173.73 s | 195.78 s | stop |
| 4 | 82,805 | 82,734 | 71 | 1.38 ms | 6.81 s | 19.28 s | stop |
| 5 | 82,873 | 82,802 | 71 | 1.20 ms | 6.83 s | 50.98 s | stop |
| 6 | 82,941 | 82,870 | 71 | 1.39 ms | 6.86 s | 38.49 s | stop |

The larger suffix in request 3 came from the returned inspection output. Requests 4–6 show fast first-token latency with only 71 new tokens, while total time still includes generation. These are internal OMP tool-cycle requests, not three distinct user follow-up samples. No process restart occurred in this subset; the later restart test is recorded below.

The OMP session executed the requested read-only Turn 1 tool calls, but ended normally after several reasoning-only responses without a visible user-facing findings/plan message. The session transcript is `/tmp/m31-product-omp-eight-turn-retry/2026-10-13T03-43-31-875Z_01a1335b-ba23-75f2-91d3-5cecef8df00e.jsonl`.

The resumed OMP session then received two additional user turns. It issued read/grep tool calls but made no repository edits and exited normally after reasoning-only output. Their first actionable tool calls arrived 195.6 s and 142.7 s after the respective user messages, despite short MIInfer first-token times on one request. Thus the 15-second product gate fails at the agent level.

| Request | Prompt | Reused | New suffix | Disk restore | TTFT wall | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 82,875 | 82,734 | 141 | 2.999 s | 12.61 s | 136.74 s |
| 11 | 86,030 | 82,872 | 3,158 | 2.960 s | 131.67 s | 142.62 s |
| 12 | 87,471 | 86,027 | 1,444 | 0 | 67.24 s | 67.24 s |

The request 7 disk restore happened while MIInfer itself stayed running; it did not prove process-restart recovery. A persistent checkpoint at 85,482 tokens existed before request 11, but request 11 matched only 82,872 tokens. The cause of that prompt mismatch remains unknown.

Turn 4's `git apply` tool action arrived 138.2 s after its user message. It applied the already-built two-file fix to the isolated OMP worktree; `git diff --check` passed. The resulting change touched only `src/prefill_v2/model.cpp` and `tools/miinfer_cli.cpp` (43 insertions, 21 deletions). It has not yet been committed or built in that worktree.

After Turn 4, the idle candidate server was stopped cleanly and restarted. The saved checkpoint at 86,252 tokens was present. OMP resumed and its first post-restart request restored that persistent snapshot: 86,466 prompt tokens, 86,252 reused, 214 new suffix, 2.872 s restore, 23.76 s TTFT, and 41.25 s total. The OMP bash tool call started 41.4 s after the Turn 5 user message. This proves state continuity across an MIInfer process restart, but misses the 15-second useful-action target. A newer 86,297-token checkpoint existed but did not match this prompt; why the prompt diverged remains open.

Turn 6 ran additional bash checks through OMP. Its first tool action arrived 34.2 s after the user message. MIInfer request 9 reused 87,239 tokens, prefetched 3 tokens, and reached first token in 0.94 s; that server TTFT is not the OMP user-to-action latency. The session still has not completed eight turns or a validated coding outcome.

### Final turns and verification retry

The resumed session completed another concrete Turn 7 action through MIInfer. It compared each edited worktree blob with commit `95516b7`, checked the original base blobs, and ran `git diff --check`. Both edited files matched the compiled candidate commit byte for byte; only `src/prefill_v2/model.cpp` and `tools/miinfer_cli.cpp` were modified. The first bash action began 53.7 seconds after that user prompt.

The next prompt (the eighth user prompt in this resumed sequence) produced literal tool-call markup and exited without dispatching the requested verification. I retried with a simpler prompt; MIInfer dispatched the bash command successfully, `git diff --check` passed, the two expected paths were listed, and `95516b7` resolved to `95516b722493b5bada1f3d4eb726c8e4b1f4dd9b`. The retry's first tool action began 33.8 seconds after submission. OMP then returned a concise summary, but the original eighth prompt did not complete as requested. Count this as seven completed prompts plus one failed dispatch and a successful retry, not as a clean eight-turn pass.

The retry transcript is in `/tmp/m31-product-omp-eight-turn-retry/2026-10-13T03-43-31-875Z_01a1335b-ba23-75f2-91d3-5cecef8df00e.jsonl`. The additional measured first-action times (53.7 s and 33.8 s) also exceed the 15-second target. This run still lacks a completed eight-turn coding outcome, and M31-PRODUCT-0001 remains not passed.

### Post-restart latency and tool-format diagnostics

Three short follow-ups were tried on the same long OMP session. With the original 4096-token profile, a `date -u +%s` bash call started 11.6 s after submission; this is a diagnostic action, not evidence of coding-task responsiveness. A useful `git diff --check` call started 16.4 s after submission. Another `git diff --stat` request returned literal `<tool_calls><invoke ...>` markup as assistant text and did not dispatch. The success and failure are both visible in the OMP transcript; do not treat the 11.6 s diagnostic as proof that the coding workflow meets the target.

The OMP profile temporarily lowered `maxTokens` from 4096 to 512. The associated request was rejected before inference (inference request count stayed at 16 while HTTP error and rejection counters each increased); no performance result can be attributed to that setting. The saved profile was restored to `maxTokens: 4096`.

OMP 18.10.2's official configuration documents `tools.format: xml` for the `<tool_calls><invoke>` in-band format. I tried it only in the isolated M31 profile. The first inference processed prompt 91,070 with a 91-token suffix in 16.46 s, generated 106 tokens, and ended with `finish_reason=stop`; it did not yield a completed OMP tool action. OMP then opened another request that stayed active without a tool result. Interrupting OMP did not stop that server-side GPU request. A 30-second container stop escalated to SIGKILL; the first model start exited 139 with a ROCm memory access fault, and a subsequent start recovered and reached ready state. The original profile was restored before further work.

After that server restart, OMP submitted the `git diff --check` recovery action. MIInfer request 4 processed a 91,221-token prompt, restored a 3,166,620,297-byte persistent session, reused 91,067 tokens, and prefilled 154 new tokens in 16.51 s; request TTFT was 20.23 s and total 22.55 s. The OMP bash action began 26.21 s after the user message and returned exit 0. A follow-up request processed 108 new tokens, took 12.97 s total, and ended normally. This verifies persistent continuation after another server restart, but misses the user-visible latency target.

The candidate container is currently ready on Machinist; GPU0 is selected and GPU2 remains idle. The profile is back to its original settings. The strongest unresolved blocker is reliable dispatch of the model's in-band XML tool calls, followed by the long suffix-prefill cost at roughly 90K context. Next action: reproduce the exact raw assistant response under the controlled OMP XML dialect and capture scanner/dispatch events and MIInfer telemetry in one short checkpoint-backed call, fixing only the dialect selection or parser path if that observation proves the mismatch.

### Explicit XML retry in progress (2026-10-13 06:03 UTC)

The isolated profile now has only one delta from `/tmp/m31-product-config.yml.before-xml-retry`: `tools.format: xml`. `maxTokens` remains 4096. OMP is running the one-command `git diff --check` request against the resumed session. The request has been active for about 22 minutes and is still in `prefill_started`; the client and MIInfer container are both live. MIInfer metrics remain at 3 inference requests, 1 active request, 262 completed prefill tokens, and 126 generated tokens. GPU0 is at 100% use and 75 C junction temperature; GPUs1 and 2 are idle. Additional request IDs have reached the server, but none has completed while this inference remains active. This is an in-flight observation, not evidence of tool dispatch or a latency result. The pre-XML profile is preserved at `/tmp/m31-product-config.yml.before-xml-retry` and must be restored after this attempt terminates.

At 06:11 UTC, the same request was still active after about 29 minutes and the completion counters were unchanged. Machinist's socket table showed one established connection to the MIInfer API, forwarded by the local SSH tunnel; the local side showed one OMP process connected to that tunnel. No separate API client explains the newer `received` request IDs. Their cause remains unknown, and they are not inference completions.

At 06:18 UTC, timestamped container logs still showed request 8 in `prefill_started`, beginning at 05:42 UTC (about 36 minutes earlier). There was no completion record; counters remained at 3 requests, 1 active, 262 completed prefill tokens, and 126 generated tokens. GPU0 remained at 100% use and 74–75 C junction temperature. This is consistent with the earlier roughly 48-minute cold-prefill observation, but the current request's prompt size and prefix-reuse count remain unreported until completion.

### Explicit XML dispatch and warm follow-ups (2026-10-13 06:23 UTC)

The long XML-format request completed. MIInfer request 8 processed the full 95,162-token prompt with no prefix reuse; prefill took 2,222.68 s (37 min 03 s), TTFT wall time was 2,223.07 s, and it generated 48 tokens. OMP parsed the in-band XML call, ran `git diff --check`, and received exit 0. The first useful action began 37 min 07 s after submission, so this cold format-transition request fails the responsiveness target. It also confirms the change from auto to XML invalidated the reusable prompt prefix.

With XML kept enabled and the prefix warm, six useful repository follow-up actions ran in the same OMP session:

| Action | Prompt tokens / reused | MIInfer TTFT wall | User message to tool start | Result |
| --- | ---: | ---: | ---: | --- |
| `git status --short` | 95,320 / 95,236 | 8.13 s | 19.66 s | Passed; two expected modified paths |
| `git diff --stat` | 95,569 / 95,454 | 9.43 s | 13.22 s | Passed; expected two-file diff |
| `git diff --check` | 95,924 / 95,838 | 8.27 s | 15.44 s | Passed; narrowly over the target |
| `git diff --name-only` | 95,993 / 95,921 | 7.72 s | 30.35 s | Command returned both expected paths; OMP attached the result after its stop sequence |
| `git diff --check` | 96,350 / 96,262 | 8.30 s | 12.58 s | Passed |
| `git diff --check` | 96,494 / 96,430 | 7.41 s | 11.78 s | Passed |
| `curl -fsS http://127.0.0.1:28080/readyz` | 96,667 / 96,574 | 8.58 s | 14.02 s | Passed; `ready:true` |
| `curl -fsS http://127.0.0.1:28080/v1/models` | 96,882 / 96,771 | 9.39 s | 14.89 s | Passed; one model listed |

The eight user-message-to-tool-start measurements sort to 11.78, 12.58, 13.22, 14.02, 14.89, 15.44, 19.66, and 30.35 seconds; the median is 14.46 seconds. Three of eight are over 15 seconds, and one result arrived after OMP's stop sequence. Treat these as short-command smoke samples, not proof that real coding follow-ups meet the target reliably. OMP's transcript records the tool calls and results in `/tmp/m31-product-omp-eight-turn-retry/2026-10-13T03-43-31-875Z_01a1335b-ba23-75f2-91d3-5cecef8df00e.jsonl`.

The original profile was restored after the cold request, then XML was re-enabled to preserve the new prompt prefix for warm follow-ups. The active profile currently has `tools.format: xml`; the original auto-format file remains backed up at `/tmp/m31-product-config.yml.before-xml-retry`. The OMP setup guidance was updated in pushed commit `3ab76af` to include this setting and the tested 4096-token output limit. The latest processed prompt was 97,002 tokens, leaving 1,302 tokens under the configured 98,304-token context; capacity for a further eight-turn sequence is not proven. `/v1/models` reported the generic id `model`, while the OMP profile uses `qwen3.8-27b`; the live OMP request still succeeded, but the id mismatch should be checked before claiming general client compatibility.

## Evidence locations

- OMP transcript: `/tmp/m31-product-omp-rpc-eight-turn/2026-10-13T01-31-21-420Z_01a132e2-b7cc-744c-b0ec-6bd423af3459.jsonl`.
- Raw RPC frames (private context/tool data; local only): `/tmp/m31-product-omp-rpc-eight-turn-resume.frames.jsonl`.
- MIInfer logs and telemetry: Machinist container `m31-product-20261012retry180`; mounted checkpoints at `/home/machinist/m31-product-campaign/20261012T163632Z/stop90/sessions/`.
- Relevant context bundle: `/tmp/m31-product-agent-task-context.md`.
- The interrupted final agent request completed on the server as request 30 (95,516 prompt tokens, 95,431 reused, 13,681.8 ms TTFT wall, `finish_reason=tool_calls`); OMP was stopped before that tool action could be dispatched or returned as a result.

### Machinist and endpoint recheck (2026-10-13 06:38 UTC)

The live host mapping confirms the selected card: GPU0 is MI50 BDF `0000:04:00.0`, unique ID `0x21678e17348c2f7`. GPU1 is BDF `0000:85:00.0`; GPU2 is BDF `0000:88:00.0`. All three reported 0% GPU use. VRAM use was 29,389,008,896 B on GPU0, 493,998,080 B on GPU1, and 10,919,936 B on GPU2. The earlier host note that labeled GPU1's BDF as the Machinist MI50 is stale; the current report's GPU0 pin is correct.

The `m31-product-agent-candidate` container was running its server process for about 1h04m. The host's existing SSH tunnel on `127.0.0.1:28080` reached `/readyz` (`ready:true`); `/v1/models` returned the generic ID `model`. The isolated Oh My Pi `m31-product` profile has `tools.format: xml`; the global default remains `auto`. No OMP process was running during this check. This verifies server readiness and the intended profile configuration, not an additional agent turn.
