# EXP-0350 — Interactive serving and live continuation

Status: RETEST — implementation candidate, correctness investigation in progress.

## Hypothesis

Combining H/I resident wide prefill with the existing Mx MMV and attention
decode route improves interactive decode; offset-aware full-layer chunks avoid
the whole-prompt fast-path restriction; exact live-frontier reuse eliminates
repeated prefix processing in a single Pi session.

## Baseline

Commit `1c38f1ea44ed`, plus the pre-existing uncommitted wide-tail scheduler
change. Historical H/I P512 numbers are benchmark-specific. The user's 3991
and 9317-token Pi requests measured 60.76 and 89.01 prompt tok/s and 1.21 and
2.08 decode tok/s. These are observations, not a matched A/B baseline.

## Candidate

- `m25_interactive`: H/I flags plus Mx MMV and attention decode reuse.
- Full-layer-major chunks take an absolute base position and preserve state.
- One experimental GPU checkpoint at the latest B512 wide-prefill boundary.
  It snapshots recurrent state/history and compact per-head KV entries, then
  restores that boundary before replaying a matching request suffix. It never
  resumes from scalar-decoded state.
- Explicit wall-time output metrics; graph capture included in step latency.
- Same-process `run --check-session` append-versus-replay regression check.

## Environment

MI50/gfx906, 34,342,961,152 bytes VRAM; Qwen3.8-27B-Q4_K_M GGUF at the
existing local model path. GCC 16.2.1, HIP Clang 20.0.0; installed HIP reports
7.1.52802. Runtime context capacity 16384. No clock changes made. Initial
idle telemetry: SCLK/MCLK 925/350 MHz, edge 32 C, 225 W power cap.
These checks are not clock-qualified throughput comparisons.

Interactive allocation at 16K: 19,479,343,444 bytes; free VRAM after loading:
14,050,918,400 bytes. The earlier user run used a different 64K configuration,
so its allocation is not an isolated measurement of the decode candidate.

## Correctness and results

Release compilation and `openai-api-host` pass. At a 512-token seed, the
one-token handoff passes: 512 retained tokens plus 152 new tokens produce the
same four output tokens as reset/replay. It measured `2533.97 ms` seed prefill,
`5561.72 ms` append prefill, `7918.79 ms` replay prefill and `21.64 tok/s`
decode. A four-token generated history fails: the retained count is correct
(515) but append outputs `93530,2319,13,220` while reset/replay outputs
`271,760,1414,488`. This isolates the issue to mixing scalar decode state with
the later wide-prefill batch, rather than offset or frontier accounting.

The check now reports exact output tokens and returns failure normally. The
first failure used an uncaught exception and started a large core dump; the
test process was stopped to release GPU memory. CLI exceptions now report an
error and exit normally. The unsafe live-frontier mechanism was replaced by a
B512 checkpoint/replay implementation. With context 1024, exact-token replay
passed at 512 and 640 prompt tokens for one- and four-token seeds. At 640 it
reused the 512-token checkpoint and replayed 280/283 suffix tokens; append
prefill was 9.17/9.29 s versus 11.44/11.69 s reset replay. This is a
correctness result at idle-clock conditions, not a throughput claim.

The interactive Mx scalar-versus-wide diagnostic was enabled for recurrent
layer zero at the failing handoff. Across B512 and B128 checks it measured
state maximum absolute error `0.00189–0.00230` and convolution-history error
`0.00269–0.00702`; QKV error was `0.00702`, gate error `0.22002`, and final
layer-zero output error `2.19–5.86`. The state handoff is therefore present
and positioned correctly, but scalar and wide numerical contracts are not
token-equivalent once their outputs pass through later layers. A cache cannot
claim stateless replay equivalence merely by preserving KV/GDN state.

The interactive preset passed the real P512 continuation check:

```text
same_process_p512_check=PASS first_token=13477(brown)
continuation_token=37550 first_prefill_ms=2312.73
continuation_ms=62.5976 repeat_prefill_ms=2366.47
```

An idle-clock P512/TG128 serving screen measured `2238.74 ms` / `228.70 tok/s`
prefill and `8218.83 ms` / `15.57 tok/s` decode. It is not a qualification
result because the device was not clock-locked. The exact 3991-token repeated
fox prompt completed with zero generated tokens in `36944.76 ms` / `108.03`
prompt tok/s. Its schedule is `512 x 7 + 384 + 23`; this confirms that the
full-layer-major chunks accept nonzero absolute positions. It does not
establish parity with real Pi text or a clock-qualified long-prompt baseline.

An HTTP SSE smoke request with 14 prompt tokens generated four content chunks.
The server recorded `745.296 ms` prefill, `19.043 ms` graph capture,
`749.793 ms` receipt-to-first-content wall time, `16.84 tok/s` steady decode
and `928.23 ms` total request wall time. The Prometheus wall-time counters
matched the log. It was a loopback smoke at idle clocks, not a Pi benchmark.

## Decision

RETEST. Checkpoint replay is now exact through the 3991-token case. At 8K
capacity, 3991-token prompts reused 3584 tokens and passed for one- and
four-token seeds; append prefill took 5.97/5.90 s versus 23.89/25.16 s for
reset replay. Keep HTTP-session reuse off by default until 8192/16000 cases
and real Pi tool-call request serialization are validated.

## Reproduction

See [interactive-serving.md](../docs/interactive-serving.md) for the exact
command, preset and metric definitions. Run correctness before any repeated
performance comparison. Pi acceptance includes structured tool-call round
trips, disconnect/cancellation, mismatch replay and real content-delta TTFT.

## Current HEAD recheck — 2026-09-19

Re-ran the existing `--check-session` against current HEAD `3165479` on the
MI50 with the Q4_K_M artifact and `MIINFER_PRESET=m25_interactive`, context
16384, and experimental-context enabled. The harness checks one- and
four-token generated seeds and compares appended output tokens with a fresh
full replay at all requested lengths. All ten cases passed:

| prompt | generated seed | reused tokens | new tokens on append | append ms | replay ms | result |
|---:|---:|---:|---:|---:|---:|---|
| 512 | 1 | 512 | 152 | 5568.21 | 8057.47 | PASS |
| 512 | 4 | 512 | 155 | 5852.23 | 8283.12 | PASS |
| 640 | 1 | 512 | 280 | 9461.33 | 12190.8 | PASS |
| 640 | 4 | 512 | 283 | 9957.15 | 12255.4 | PASS |
| 3991 | 1 | 3584 | 559 | 5864.27 | 24619.6 | PASS |
| 3991 | 4 | 3584 | 562 | 6172.41 | 25348.9 | PASS |
| 8192 | 1 | 8192 | 152 | 6290.92 | 57241.7 | PASS |
| 8192 | 4 | 8192 | 155 | 6354.21 | 56372.6 | PASS |
| 16000 | 1 | 15872 | 280 | 12790 | 134990 | PASS |
| 16000 | 4 | 15872 | 283 | 12506 | 134837 | PASS |

The timing run was not clock/thermal qualified, and the 16000-token one-seed
case showed a substantial seed-prefill timing outlier. Treat these as
correctness and savings-shape evidence only. At this point the implemented
single-checkpoint length/seed matrix was complete; later lifecycle and HTTP
checks are recorded below. Multi-checkpoint lookup remains unimplemented.

## Cancellation/failure invalidation and tool-turn reuse — 2026-09-19

The current `generate()` path retained the old checkpoint identity after a
cancelled or throwing appended generation. It now clears the stored prefix
identity on either outcome while retaining the allocated GPU buffers. This
makes the next request replay rather than restoring a checkpoint associated
with an interrupted request.

Built `miinfer` in `build/mi50-release` and ran the live-model `--check-session`
at context 1024. The 512/640 one- and four-token append/replay cases passed.
Then the same diagnostic cancelled an appended generation, injected a
generation exception, supplied a mismatching prefix, and explicitly reset the
engine. All invalidation assertions passed:

```text
session_invalidation=PASS cause=cancellation reused_before=512 reused_after=0
session_invalidation=PASS cause=generation_failure reused_after=0
session_invalidation=PASS causes=mismatch,reset reused_after=0
```

An HTTP smoke used one server process with `MIINFER_PRESET=m25_interactive`,
`MIINFER_SESSION_REUSE=1`, and context 1024. The first OpenAI request contained
a 700-word user prompt and a `read_file` tool schema; it returned HTTP 200 with
930 prompt tokens. The second request extended that conversation with an
assistant `read_file` tool call, a tool result, and a follow-up user message;
it returned HTTP 200 with 1001 prompt tokens. Server telemetry showed:

```text
request 1: reused=0 new=930 cache_hit=false
request 2: common_prefix=512 reused=512 new=489 cache_hit=true
```

This proves request parsing, tool-call/result ChatML serialization, and exact
512-token prefix matching through the real server path. These were functional
checks at the current device state, not throughput qualification. Multi-session
isolation and persistent model/runtime contract identifiers remain
unqualified; cache stays experimental and single-session only.

## Decision update

**RETEST.** The 512–16000 length/seed matrix, cancellation/failure/mismatch/
reset invalidation, and a serialized HTTP tool-result turn all pass. Keep reuse
experimental until real Pi session identity/tool-loop behavior is qualified.
Cold P512 remains a separate track; none of this changes its benchmark claim.

## Current HEAD 4K append and real Pi tool loop — 2026-09-19

### Same-prompt append versus replay

An OpenAI-compatible two-request tool conversation was run on one server at
context 8192. The second request was then sent unchanged to a fresh server
process for the replay control. Both requests tokenized to 3705 tokens:

| route | total tokens | reused | processed | prefill ms | cache hit |
|---|---:|---:|---:|---:|---|
| checkpoint append | 3705 | 3584 | 121 | 7600.32 | yes |
| fresh-process replay | 3705 | 0 | 3705 | 26193.7 | no |

These are one sample per route, with no continuous clock/thermal capture. They
show that 96.7% of the identical request's prompt tokens were not reprocessed;
the raw latency readings are diagnostic only (about 3.45x in this pair), not a
qualified serving-speed claim.

### Pi coding-agent tool cycle

Pi CLI `0.85.1` was pointed at the MIInfer server through an isolated temporary
`PI_CODING_AGENT_DIR`; the existing Pi settings and credentials were not
changed. The run explicitly attached this repository's `AGENTS.md`, enabled
the built-in `read` tool, and asked Pi to read a local fixture containing the
exact strings `MI50` and `amber-cedar`. Pi made the tool call, consumed the
result, and returned both exact values.

With `MIINFER_PRESET=m25_interactive`, `MIINFER_SESSION_REUSE=1`, context 8192,
and one MIInfer server process, the server logged:

| request | prompt tokens | reused prefix | new prefill | cache hit | outcome |
|---|---:|---:|---:|---|---|
| initial Pi turn with project context | 6053 | 0 | 6053 | no | tool call emitted |
| Pi tool-result continuation | 6129 | 5632 | 497 | yes | correct summary returned |

This is a real Pi tool round trip with project context, rather than a synthetic
OpenAI client transcript. It demonstrates that the agent's serialized tool
history preserves a long exact prefix across requests. These timings were not
captured as a controlled performance comparison. Real interactive Pi session
resume/restart, cancellation, and full multi-turn coding tasks remain open.

## Current build exact checkpoint recheck — 2026-09-20

After the M27 cold-contract stop, reran the built-in append-versus-replay
matrix from the current build (`ab88603` plus profiling-only changes), with
`MIINFER_PRESET=m25_interactive`, context 16384, and experimental context
enabled. Each case compared generated token IDs against a fresh full replay.
All ten cases passed:

| prompt tokens | generated seed | reused | appended prefill tokens | append ms | full replay ms |
|---:|---:|---:|---:|---:|---:|
| 512 | 1 | 512 | 152 | 5467.92 | 8059.73 |
| 512 | 4 | 512 | 155 | 5745.77 | 8258.19 |
| 640 | 1 | 512 | 280 | 9387.86 | 12056.7 |
| 640 | 4 | 512 | 283 | 9317.09 | 12062.6 |
| 3991 | 1 | 3584 | 559 | 5842.51 | 25038.7 |
| 3991 | 4 | 3584 | 562 | 6218.90 | 24979.5 |
| 8192 | 1 | 8192 | 152 | 6211.62 | 56197.9 |
| 8192 | 4 | 8192 | 155 | 6244.57 | 56329.1 |
| 16000 | 1 | 15872 | 280 | 11960.6 | 134188 |
| 16000 | 4 | 15872 | 283 | 12322.6 | 134578 |

The same run passed cancellation, injected generation failure, prefix mismatch,
and explicit reset invalidation. The raw run is
`/tmp/m27-context-smoke/session-check-current.log`. This rechecks correctness,
not serving performance: no continuous telemetry was captured and the device
was not clock-qualified for these long cases. It closes the single-checkpoint
exactness gate for the current runtime, while real Pi restart/resume and
cross-session isolation remain open.

## Execution-contract version tag — 2026-09-20

Each in-memory checkpoint now records execution-contract version 1 beside the
exact token prefix and hybrid recurrent/history/KV state. Restore fails loudly
on a version mismatch; the runtime falls back to full replay before attempting
restore. The full 512–16000-token matrix above ran immediately before this
metadata guard. After the guard was added, the 512/640 one- and four-token
cases plus cancellation, generation-failure, mismatch and reset invalidation
were rerun at context 1024; all passed. Raw output:
`/tmp/m27-context-smoke/session-check-version.log`.

## M27 prefix-reuse closure — 2026-09-20

**KEEP for opt-in, single-session serving.** The current build passes exact
append-versus-replay token checks at 512, 640, 3991, 8192, and 16000 tokens
with one- and four-token generated seeds; it also passes cancellation,
generation-failure, mismatch, reset, and execution-contract-version checks.
The real Pi 0.85.1 tool-call/result cycle reused 5632 of 6129 tokens on the
second request. A repeated 3705-token serialized request reused 3584 tokens
and processed 121. Together these establish functional multi-turn tool-history
reuse and that large repeated prefixes avoid full replay. The timing data is
not a serving-speed qualification.

P1/P2/P3 are deferred: the measured agent flow is linear and already reuses
its latest exact B512 checkpoint. No observed request has missed a useful
older checkpoint. At 16K one checkpoint costs approximately 1.05 GiB of
additional GPU memory, so retaining multiple snapshots trades context
headroom for an unmeasured branch benefit. Add a small multi-checkpoint cache
with longest-prefix scan only after a captured real-agent trace shows a
divergent request whose reusable prefix predates the latest checkpoint. A
radix tree has no demonstrated lookup need at the current one-entry scale.

This closes the M27 prefix-reuse objective at the existing exact single-
checkpoint scope; it does not qualify reuse across process restart or across
sessions. Keep `MIINFER_SESSION_REUSE=1` opt-in until those serving lifecycle
requirements are addressed. Cold P512 remains stopped per EXP-0356; M26 decode
work remains paused with its evidence intact.
