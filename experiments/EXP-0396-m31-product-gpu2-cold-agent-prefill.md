# EXP-0396 — M31 product GPU2 cold agent prefill

**Status:** RETEST  
**Milestone:** M31-PRODUCT-0001  
**Date:** 2026-10-12  
**Baseline commit:** 3991b6bf40f810b63a100c46981e16964bb5434b  
**Candidate commit:** 3991b6bf40f810b63a100c46981e16964bb5434b

## Question

Can the second MI50 on Machinist run the same large-context Oh My Pi request at the user-set 180 W cap, and return a useful tool action?

## Environment

- Host: Machinist X99. Z840 was not used.
- Active GPU: GPU 2, MI50 gfx906, PCI BDF `0000:88:00.0`, UUID `0x2152414173497dd3`.
- Idle GPU: GPU 0, MI50 gfx906, PCI BDF `0000:04:00.0`, UUID `0x21678e17348c2f7`.
- Both MI50s reported a 180 W power cap before the run.
- Model: Qwen3.8-27B Q4_K_M, SHA-256 `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.
- Image: `localhost/miinfer-dev:rocm-7.2.1`, digest `sha256:447907c697fb2c39ce9657401e354a70753d5672ad955d4ceb12de976b91224a`.
- MIInfer source: `3991b6bf40f810b63a100c46981e16964bb5434b`.
- MIInfer binary SHA-256: `6fbcd70f2e1e9ddf47f1291b688776a1ecb5193f681a8e8921907284db409a32`.
- Server settings: 98,304-token capacity, Q8 KV cache, persistent session reuse enabled.
- Thermal guard: warning at 80°C, stop at 90°C.

## Benchmark

Oh My Pi used the `m31-product-gpu2` profile with the same two context attachments and first-turn task used in the earlier GPU0 run. MIInfer processed 73,099 prompt tokens. The attachments and tool definitions made this larger than an exact 65,536-token prefix, and the attached history was not a representative coding repository context.

The local OMP command ran in print mode. Its terminal log contained only `Working...`. The inference server completed the request with `finish_reason=tool_calls`, but the OMP client did not save a resumable session before the run cutoff. The generated tool action and its execution were not verified.

## Results

| Measure | GPU2 result |
| --- | ---: |
| Prompt tokens processed | 73,099 |
| Prefill | 2,298,230 ms |
| Prefill throughput | 31.8067 tokens/s |
| First decode token | 2,298,230 ms |
| TTFT wall | 2,298,433 ms |
| Decode | 18,939.1 ms |
| Decode throughput | 12.7778 tokens/s |
| Generated tokens | 243 |
| Total request time | 2,317,290 ms |
| Reused prefix tokens | 0 |
| Persistent session restored | No |
| HTTP errors | 0 |
| Junction temperature during the measured run | 30–66°C |

The run completed without a thermal warning. The guard file also contains 80–81°C warnings from an earlier concurrent attempt, before this sample. Telemetry samples during this run occasionally exceeded the requested 180 W cap, with a sampled maximum of 230 W; treat the power readings as approximate until the cap telemetry is reconciled.

## Interpretation

GPU2 stayed cooler than GPU0 had in the earlier run, but this run did not show a latency benefit. The earlier GPU0 run measured 73,066 prompt tokens, 53.8335 tokens/s prefill, and 22.394 seconds for a cached follow-up. Only one cold-prefill sample per GPU is available, so these values do not establish a hardware winner.

### GPU0 interactive OMP run

A later interactive OMP run on GPU0 produced additional server-side evidence. It used the same MIInfer source, model and 98,304-token/Q8-KV configuration. The completed cold request contained 48,105 prompt tokens, not the 73,066 tokens from the earlier GPU0 sample:

| Measure | GPU0 interactive result |
| --- | ---: |
| Cold prompt tokens | 48,105 |
| Cold prefill | 1,118,390 ms |
| Cold prefill throughput | 43.0127 tokens/s |
| Generated tokens | 2,048 |
| Cold finish reason | `length` |
| Follow-up prompt tokens | 10,870 |
| Reused follow-up prefix | 10,782 tokens |
| Follow-up checkpoint bytes | 3,405,250,560 |
| Follow-up wall TTFT | 1,560.9 ms |
| Follow-up total request time | 3,521.8 ms |

The first follow-up request (10,782 tokens) returned `finish_reason=tool_calls` in 68.6 seconds, but it reported no prefix reuse. The next request reused those 10,782 tokens and completed in 3.52 seconds. This demonstrates fast same-server prefix reuse for this shorter follow-up. The logs do not establish that OMP executed the tool call, that state survived a server restart, or that a 64K eight-turn session worked. The cold request stopped at the generation limit before its task outcome was verified.

### OMP tool execution and follow-up gate

On 2026-10-13, a controlled OMP v18.10.2 run used the `m31-product` profile, GPU0 (`0000:04:00.0`, UUID `0x21678e17348c2f7`), the same model and MIInfer server configuration, and disabled only title generation. OMP called `bash` to run `git status --short --branch && git rev-parse --short HEAD`; the tool result was `## m31-product-agent-task...origin/m31-product-agent-task` and `c840482`. This verifies one actual OMP tool call and its returned result.

| Request | Prompt tokens | Reused tokens | Restore | Request wall | Outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| 7, initial OMP turn | 13,278 | 0 | No | 92.1 s | `tool_calls`; bash ran successfully |
| 8, OMP turn after tool result | 13,457 | 0 | No | 135.1 s | Full prefill; answer returned |

The post-tool prompt processed all 13,457 tokens again (`prefill_ms=89,514`, `cache_hit=false`, `persistent_session_restored=false`). Its end-to-end response exceeded the 15-second target. This print-mode run had no title-generation traffic, so the cache miss is not explained by the earlier title-generator queue. A separate interactive attempt did restore 14,021 tokens and reported `persistent_session_restored=true`, but its total wall time was 109.4 seconds, including about 99.7 seconds of queue wait after a cancelled duplicate request. It also missed the responsiveness target.

The controlled run's OMP transcript and request logs prove tool invocation, result delivery, and the follow-up re-prefill. They do not yet identify why the OMP follow-up prompt fails MIInfer's token-prefix match. Temperatures stayed below the 80°C warning threshold; the 90°C stop guard did not fire.

### OMP process-resume and disk-checkpoint follow-up

To isolate the resume path, I routed OMP through a temporary local HTTP capture proxy and resumed the same saved OMP session twice. The proxy retained request bodies locally for comparison; no captured prompt content is committed. OMP called `bash` to print `M31_CAPTURE_TURN` and later `M31_SECOND_TOOL`; both commands executed and their exact stdout was returned.

| Request | Prompt tokens | Reused tokens | Disk restore | TTFT | Total request | Outcome |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| 16, resumed OMP tool request | 13,628 | 0 | No | 90.6 s | 94.6 s | `tool_calls`; bash ran |
| 17, response after tool result | 13,716 | 13,628 | No | 1.71 s | 29.1 s | 88-token suffix; answer returned |
| 18, second OMP process resume | 13,785 | 13,628 | Yes, 608,965,005 bytes | 3.24 s | 6.13 s | `tool_calls`; bash ran |
| 19, response after second tool result | 13,873 | 13,785 | No | 1.69 s | 6.92 s | 88-token suffix; answer returned |

Request 18 is the first controlled evidence that the saved MIInfer checkpoint restores after the OMP client process exits: telemetry reports `persistent_session_restored=true`, a 13,628-token common prefix, 157 newly prefetched tokens, 670.9 ms restore time, and a 3.24-second wall TTFT. OMP then executed the requested tool. Request 19 reused all 13,785 tokens and prefetched only 88 new tokens. The MIInfer server remained running throughout, so this proves disk-checkpoint restore across OMP process restart, not inference-server restart or crash recovery.

Request 17's first token arrived at 1.71 seconds, although the complete 595-token answer took 29.1 seconds. Request 18 reached a tool call in 6.13 seconds total. These are one-sample results below the 15-second first-action target at 13K context; they do not yet establish a median or 64K responsiveness. This controlled extension shows cache matching works when OMP supplies an exact prompt continuation. Earlier misses remain attributable to a different prompt/session state, but the exact mismatch in those older requests was not captured.

The run used the same frozen MIInfer image/model and GPU0; GPU2 stayed idle. GPU0 returned to 35°C junction after the run. No thermal guard event occurred.

### 64K relevant-context prefill

I assembled 246,640 characters of relevant MIInfer project context: project guidance, serving/architecture/CLI docs, API and persistent-session implementation, an API test, and this experiment report. OMP submitted that context to the same server. I caught that the OMP process had started from the main checkout rather than the campaign branch and canceled it before any repository tool action or edit. The server finishes prefill after a client disconnect; it reported:

| Measure | Result |
| --- | ---: |
| Total prompt tokens | 78,852 |
| New prompt tokens processed | 65,067 |
| Prefix restored from disk | 13,785 tokens / 614,150,401 bytes |
| Suffix prefill time | 1,464,290 ms |
| Total prefill time | 1,467,320 ms |
| Configured context | 98,304 tokens |
| Headroom at this prompt size | 19,452 tokens |
| Client result | Canceled after 4 generated tokens; no useful response/tool action |

MIInfer saved checkpoint `sess_c484a78843ec60f8_p78852.miinfer`. This is evidence that the runtime processed 65,067 new, relevant repository-context tokens and retained headroom; it is not a completed 64K agent turn. GPU0 peaked at 76°C junction during prefill and returned to 32°C afterward. GPU2 remained idle.

The first OMP resume attempt at this context size did not reuse that checkpoint. Request 21 processed 78,970 prompt tokens but restored only the older 13,785-token checkpoint, then prefetched 65,185 tokens in 1,472,020 ms. The OMP client had a 3-minute limit and exited with `Deadline exceeded`; the server later generated 85 tokens with `finish_reason=tool_calls`, but the OMP transcript contains no corresponding tool execution or result. Request 21 wrote checkpoint `sess_80a0fab81090393c_p78970.miinfer`. The reason the 78,852-token checkpoint did not match after OMP resume is not yet identified; prompt-token divergence after 13,785 tokens is the current blocker.

### OMP plural tool-call parser diagnosis

The current GPU0 serving container's generated `build_config.hpp` identifies source commit `7bd18de66948`; its running executable hash is `e239b48a1fbeaec46b865a247d0500e14ea8ef5f2a9279fafd579728fbbbae0a`. This corrects the earlier attribution of the current GPU0 service to `3991b6b` and binary hash `6fbcd70f...`; those values do not identify the executable now serving OMP requests.

At `7bd18de`, `append_generated_xml_calls` was changed to parse functions only inside singular `<tool_call>...</tool_call>` wrappers. OMP request 8's transcript contains two function calls enclosed by the plural `<tool_calls>...</tool_calls>` wrapper, followed by MIInfer `finish_reason=stop`; OMP treated the markup as assistant text and ran neither call. The pushed `ee911e2` change adds plural-wrapper parsing and a regression assertion for the exact two-call shape.

I copied only `src/openai_api.cpp` and `tests/openai_api_test.cpp` into a temporary Machinist worktree based on the runtime source, then built and ran `miinfer-openai-api-test` with the pinned ROCm image. Result: `openai API host test passed`. This verifies the parser fix with the serving toolchain. The earlier 65K resume cache miss remains a separate issue: even with valid tool parsing, request 21 replayed 65,185 tokens and exceeded the latency target.

I then built the `miinfer` executable in the same pinned image from the existing `7bd18de` build tree with only the parser and test source files updated. The replacement binary SHA-256 is `1dcfd59d73ad4949e26135b445c11d5dd3a9aff05ffac0fa625a8235c0e28c84`; its strings include the plural wrapper markers. The serving container was restarted, `/healthz` returned `{"status":"ok","ready":true}`, and GPU0 was idle at 25°C junction after load.

A live OMP v18.10.2 print-mode smoke on the rebuilt server asked for the same two read-only tools seen in request 8. The OMP transcript `01a132d7-90b6-73ac-8830-9cd945dc5698` records two assistant `toolCall` entries and two `toolResult` entries (bash and read), followed by a grounded summary. MIInfer requests 2 and 3 processed 9,248 and 11,066 tokens with no session reuse. Temperatures reached 75°C junction; GPU2 remained idle. This closes the short-context tool-parsing blocker, not the 64K persistence gate.

After restarting the MIInfer server, I resumed the old high-context OMP session. OMP sent 13,321- and 15,542-token prompts; both requests reported zero reused tokens and `persistent_session_restored=false`. This resume path did not carry the 65K context into the restarted client session, so it cannot prove high-context restore. A single long-lived OMP RPC session is the next candidate: keep the full conversation in memory through eight prompts, then restart only MIInfer after turn four and inspect the disk-restore telemetry on the next message.

### Long-lived OMP RPC high-context turn

I launched Oh My Pi v18.10.2 in its JSONL RPC mode and sent the 246,640-character relevant-context bundle (SHA-256 `2ed2d7b915f45a1951909a5c6664fb09f2e0813bbbad07ab1d5e4dcc80350478`) with a one-turn inspect-only task. MIInfer request 7 fully prefetched 74,992 prompt tokens with zero reused tokens. Configured capacity was 98,304, leaving 23,312 tokens of initial headroom. The turn took 2,132.78 seconds end to end and executed seven bash calls with seven returned tool-result messages; it made no file edits.

The same OMP RPC session then issued requests 8–13. MIInfer reused the complete prior prefix on every request: prompt size grew from 79,897 to 84,654 tokens, with each request matching the previous prompt length. The final context headroom after this tool exchange was 13,650 tokens. MIInfer saved checkpoint `sess_ecbc6e016ddd265f_p84654.miinfer` (2,954,811,733 bytes). The OMP session transcript is saved locally as `01a132e2-b7cc-744c-b0ec-6bd423af3459.jsonl`; the raw RPC frame capture is local-only because it contains the full prompt and tool outputs.

The RPC driver received `session_settled` and closed OMP normally after turn one because its parent stdin reached EOF. This did not corrupt the saved transcript or MIInfer checkpoint, but it means turns two through eight still need to resume from that saved session. The next check is whether RPC resume preserves the 84K prompt and restores the checkpoint; if it does, continue with small tool outputs to protect remaining context headroom.

On GPU2, neither a fresh OMP process nor a resumed OMP transcript demonstrated a fast follow-up. Two requests entered prefill and remained active for about three minutes before I stopped the GPU2 container, which exited with code 137. The resumed transcript ID was `01a131bf-1af4-734d-9113-f0da8b510ec9`. An exact no-session replay of the original print-mode request also entered prefill and stayed active for one minute before I stopped it. Disk checkpoint restore did not produce a fast response in these attempts. The eight-turn task and a cached follow-up on GPU2 remain unproven.

### MIInfer service recovery after forced stop

After stopping the repeated cold prefill, the first forced container stop ended the process with exit code 139 and a ROCm memory-access fault during model reload. A subsequent `podman start` loaded the same model successfully without changing the container configuration. The API returned `{"status":"ok","ready":true}`; ROCm reported MIInfer using GPU0 only, with GPU2 idle. This establishes service recovery, not checkpoint restoration across an MIInfer server restart.

### Fresh OMP session deadline

A fresh OMP session (`01a13284-0eda-7421-9944-59683def7297`) submitted the 246,640-character relevant-context bundle and the read-only first-turn task. The client was configured with a 45-minute maximum. At that deadline OMP recorded `Deadline exceeded`, with no assistant content or tool result. MIInfer request 3 had reached `prefill_started`; after the client exited, the server still reported one active request, GPU0 at 100% utilization, and 67°C junction. The thermal watchdog remained active, with no warning or stop. Requests 4 and 5 had been received but not completed. The OMP transcript is retained locally under `/tmp/m31-product-8turn-sessions/`; its prompt contents are not committed.

Request 3 later completed at 02:33:56 CEST: 78,142 prompt tokens, 78,142 newly prefetched, zero reused, 2,747,940 ms prefill at 28.44 tokens/s, and 2,757,910 ms total. MIInfer generated 130 tokens with `finish_reason=tool_calls` and saved checkpoint `sess_3cb1819307571240_p78142.miinfer` (2,739,733,397 bytes). The server did not cancel the request. OMP had already exited about one minute earlier, so its transcript still has no tool execution or result.

The same OMP session was then resumed with a longer client deadline. Request 8 entered prefill at 02:35:15 CEST and completed at 03:01:10 CEST: 78,260 prompt tokens, zero reused, 78,260 newly prefetched, 1,548,130 ms prefill at 50.55 tokens/s, and 1,555,860 ms total. `persistent_session_restored=false`. GPU0 peaked at 76°C junction; GPU2 stayed idle. OMP received 98 tokens, but the assistant returned literal `<tool_calls>...` markup as text with `finish_reason=stop`; the OMP transcript has no tool execution or result. The high-context checkpoint was present, but the resumed prompt did not match it and the intended inspection action did not run.

## Decision

**RETEST.** OMP tool execution and disk-checkpoint restore across OMP client-process restart are demonstrated at 13K context: the restored turn reached a real tool call in 6.13 seconds. A fresh long-lived RPC session has now processed 74,992 prompt tokens, executed seven real bash calls/results, and reused its full growing prefix through 84,654 tokens. Its disk checkpoint is 2.95 GB. The RPC transcript was saved after turn one; next verify that resuming this saved session retains the 84K context and MIInfer state. If it does, continue the coding task through eight turns, checkpoint after turn four, restart MIInfer at that boundary, and measure the restored follow-up action. If it again resumes at 13K, the exact client-persistence gap remains the blocker. Keep GPU2 idle.

## Evidence

- Machinist campaign logs: `/home/machinist/m31-product-campaign/20261012T163632Z/gpu2/`.
- Completed inference request: request ID 5 in the `m31-product-gpu2` container log.
- Temperature and power samples: `gpu2/telemetry.csv`.
- Local OMP print-mode capture: `/tmp/m31-product-gpu2-first-turn.log`.
- GPU0 interactive server requests: request IDs 2, 13, and 15 in container `m31-product-20261012retry180`; telemetry in `retry180/telemetry.csv`.
- Controlled OMP transcript: `/tmp/m31-product-omp-controlled-20261013/2026-10-12T22-16-31-033Z_01a13230-5639-7565-9945-6f570fd98e46.jsonl`; first-turn output: `/tmp/m31-product-omp-controlled-20261013-first.log`.
- Controlled OMP server requests: IDs 7 and 8 in `m31-product-20261012retry180`; telemetry in `retry180/telemetry.csv`.
- OMP process-resume transcript: same session ID `01a13230-5639-7565-9945-6f570fd98e46`; server requests 16–19 in `m31-product-20261012retry180`; exact request bodies in local-only `/tmp/m31-omp-api-captures.jsonl` (contains private system/tool prompt text; do not publish).
- Relevant-context bundle: `/tmp/m31-product-agent-task-context.md` (246,640 characters; local temporary artifact). Server request 20 processed it and saved `sess_c484a78843ec60f8_p78852.miinfer` under the campaign `stop90/sessions` directory.
- Server request 21 attempted OMP restore of that high-context state, matched only the prior 13,785-token checkpoint, and saved `sess_80a0fab81090393c_p78970.miinfer`; its OMP client transcript has no tool result.
