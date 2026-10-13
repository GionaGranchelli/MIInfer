# EXP-0397 — Oh My Pi persistent eight-turn task on MI50

**Status:** RETEST — submitted eight user turns; the final validation turn did not settle after a bad assertion triggered repeated retries. Latency and repeatable state continuity failed the product gate.  
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

The larger suffix in request 3 came from the returned inspection output. Requests 4–6 show fast first-token latency with only 71 new tokens, while total time still includes generation. These are internal OMP tool-cycle requests, not three distinct user follow-up samples. No MIInfer process restart or disk restore has yet been tested in this retest.

The OMP session executed the requested read-only Turn 1 tool calls, but ended normally after several reasoning-only responses without a visible user-facing findings/plan message. The session transcript is `/tmp/m31-product-omp-eight-turn-retry/2026-10-13T03-43-31-875Z_01a1335b-ba23-75f2-91d3-5cecef8df00e.jsonl`. The eight-turn workload, restart restore, and follow-up usability gate remain unproven.

The resumed OMP session then received two additional user turns. It issued read/grep tool calls but made no repository edits and exited normally after reasoning-only output. Their first actionable tool calls arrived 195.6 s and 142.7 s after the respective user messages, despite short MIInfer first-token times on one request. Thus the 15-second product gate fails at the agent level.

| Request | Prompt | Reused | New suffix | Disk restore | TTFT wall | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 82,875 | 82,734 | 141 | 2.999 s | 12.61 s | 136.74 s |
| 11 | 86,030 | 82,872 | 3,158 | 2.960 s | 131.67 s | 142.62 s |
| 12 | 87,471 | 86,027 | 1,444 | 0 | 67.24 s | 67.24 s |

The request 7 disk restore happened while MIInfer itself stayed running; this does not prove process-restart recovery. A persistent checkpoint at 85,482 tokens existed before request 11, but request 11 matched only 82,872 tokens. The cause of that prompt mismatch remains unknown. The OMP worktree is still clean at `29fbc26`; turn four, an MIInfer restart, and a correct coding outcome have not been reached.

## Evidence locations

- OMP transcript: `/tmp/m31-product-omp-rpc-eight-turn/2026-10-13T01-31-21-420Z_01a132e2-b7cc-744c-b0ec-6bd423af3459.jsonl`.
- Raw RPC frames (private context/tool data; local only): `/tmp/m31-product-omp-rpc-eight-turn-resume.frames.jsonl`.
- MIInfer logs and telemetry: Machinist container `m31-product-20261012retry180`; mounted checkpoints at `/home/machinist/m31-product-campaign/20261012T163632Z/stop90/sessions/`.
- Relevant context bundle: `/tmp/m31-product-agent-task-context.md`.
- The interrupted final agent request completed on the server as request 30 (95,516 prompt tokens, 95,431 reused, 13,681.8 ms TTFT wall, `finish_reason=tool_calls`); OMP was stopped before that tool action could be dispatched or returned as a result.
