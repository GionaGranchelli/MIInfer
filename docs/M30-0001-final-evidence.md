# M30-0001 final evidence

Status: implementation correctness is focused-green; canonical performance qualification is incomplete and is not a pass.

## Implementation evidence

- Candidate branch: `m30/exact-prefix-kv-reuse`
- Candidate SHA: `9aaa8a128ac29cf5d49e8bcd44ac0e66de99665a`
- Contract SHA: `1b6c47e45b6a18e5e74a88947efdb8099bc15b1d`
- Focused lifecycle test: `M30-0001 exact-prefix reuse: PASS zero_suffix_twice=PASS suffix=PASS mismatch_fallback=PASS fresh_session=PASS`
- Runtime probe: cold request 13 tokens; exact continuation reused 13 with zero suffix; extension reused 13 and executed 17 suffix tokens; repeated extension reused 30 with zero suffix; response SHA matched.

The implementation records both GQA and GDN ownership. A canonical reuse hit reported both `gqa_kv_reused_tokens` and `gdn_checkpoint_position`, so this is not KV-only evidence.

## Canonical run result

Remote artifacts:

- `results/m30-0001-agent-workload.json`
- `results/m30-0001-server-128k.log`

The workload completed 40 logical turns, but emitted 82 completed requests rather than the M30-0000 baseline's 86. The candidate result also reports `all_turns_minimum_generation_gate=true` and `all_turns_visible_answer_gate=false`; the baseline artifact reports the same visible-answer gate as false, so that gate is not treated as the primary discriminator here.

Candidate server-log totals across 82 latency records:

| Metric | M30-0000 baseline | M30-0001 observed |
|---|---:|---:|
| Requests | 86 | 82 |
| Logical prompt tokens | 3,651,780 | 3,785,918 |
| New/physically prefetched tokens | 1,582,245 | 1,790,897 |
| Reused prefix tokens | 2,069,535 | 1,995,021 |
| Prefix tokens replayed | baseline field unavailable | 1,732,197 |
| Prefill wall sum | 10,810,661.9 ms | 12,307,081.7 ms |
| Request wall sum | 12,918,848.7 ms | 14,249,068.7 ms |
| Maximum checkpoint bytes | 8,748,793,856 | 8,748,793,856 |

The candidate log contains genuine suffix-only reuse, including request 247: 81,749 reused prefix tokens, 1,336 suffix tokens executed, zero prefix replay, and a 1.56 ms restore. It also contains legitimate full-replay fallbacks, including request 244: 81,749 tokens prefetched and replayed.

## Qualification boundary

The 82-request run is evidence that the mechanism operated, but it is not a valid apples-to-apples 86-request performance qualification. The candidate and baseline tool-call trajectories diverged at multiple logical turns; therefore the higher candidate prompt and wall totals cannot be attributed solely to the reuse implementation. No M30-0001 success-gate claim is made from this run.

The first prompt already differed before any candidate reuse hit: the candidate recorded 5,372 prompt tokens, while the baseline recorded 5,354. The workload embeds repository excerpts in its initial context, including the first 6,000 bytes of `include/miinfer/prefill_v2/model.hpp`; that excerpt changed between the baseline and candidate commits. Consequently, later tool-call trajectory differences and the 82-versus-86 request count cannot isolate reuse behavior. A valid rerun must execute the candidate server with the frozen M30-0000 workload context (including the baseline source excerpts and tool results).

## Frozen-context rerun

The follow-up rerun used the baseline `model.hpp` excerpt hash
`a690883a2f2737de3d0a3ec844fd7cfb6784b992b0186b37fc5ba478b190547a` and
recorded the first prompt at 5,354 tokens, matching the corrected baseline.
Artifacts:

- `results/m30-0001-frozen2-agent-workload.json`
- `results/m30-0001-frozen2-server.log`

The run completed 40 logical turns and 82 requests. It recorded 46 reuse hits,
1,889,692 reused prefix tokens, 1,567,349 new-prefill tokens, 10,508,506.4 ms
of prefill time, 12,278,839.8 ms of request time, and 34,323,070,976 bytes of
peak VRAM. The visible-answer gate was false on turns 2, 19, and 21; the
baseline artifact also has visible-answer failures, so this is not treated as
candidate-only evidence.

This remains non-qualifying performance evidence. The candidate's cold first
turn diverged from the baseline transcript before reuse could affect it: the
third response was 820 characters versus 69 in the corrected baseline. The
resulting tool/request trajectory stayed at 82 rather than 86 requests. The
rerun proves live GQA+GDN reuse and memory behavior under the frozen initial
context, but it does not support an apples-to-apples wall-clock or token
reduction claim against M30-0000.

## Fixed-transcript 86-request replay

To remove model-response trajectory drift from the comparison, the corrected
M30-0000 assistant/tool transcript was replayed against the candidate while
advancing the request history from the frozen baseline transcript. This is a
valid 40-turn/86-request run, but it is a performance replay rather than a
second output-correctness oracle: canonical focused correctness is covered by
the lifecycle tests above.

Artifacts:

- `results/m30-0001-fixed-replay.json`
- `results/m30-0001-fixed-replay-server.log`
- `bench/m30_fixed_transcript_replay.py`

| Metric | M30-0000 baseline | M30-0001 fixed replay |
|---|---:|---:|
| Requests | 86 | 86 |
| Logical prompt tokens | 3,651,780 | 3,631,536 |
| Physically prefetched tokens | 1,582,245 | 1,574,774 |
| Reused prefix tokens | 2,069,535 | 2,056,762 |
| Reuse hits | baseline telemetry unavailable | 53 |
| Prefill wall sum | 10,810,661.9 ms | 10,773,544.6 ms |
| Total request wall sum | 12,918,848.7 ms | 12,868,518.2 ms |
| Generated tokens | 41,114 | 40,911 |
| Peak VRAM | 34,323,197,952 bytes | 34,323,324,928 bytes |

The fixed replay met the 40-turn/86-request and minimum-generation gates. It
reused 99.38% of the baseline's exposed repeated-prefix total and executed
1,574,774 suffix/new-prefill tokens, with 33 cold/fallback requests and 53
reuse hits. However, the measured physical-prefill reduction is only 7,471
tokens (0.47%), and total request wall reduction is 50,330.5 ms (0.39%). The
candidate also reports 20,244 fewer logical prompt tokens than the baseline
artifact. Therefore this run proves exact-trajectory telemetry and live reuse,
but does not demonstrate a meaningful canonical performance improvement. The
implementation remains focused-green; the M30-0001 canonical success gate is
not claimed as passed.

The 32 GiB Machinist 27B runtime attempt remains separately blocked by HIP out-of-memory during recurrent-layer construction and is not used as correctness evidence for the Z840 run.

## Baseline qualification boundary

The fixed replay exposed an additional limitation in the comparison itself.
M30-0000 was described as a measurement-only baseline, but its production
server was already started with session reuse enabled. The serving adapter
defaults `session_reuse` to enabled, and its existing `SessionCheckpoint`
stores/restores both GDN recurrent state/history and GQA key/value state. The
baseline artifact consequently already reports 53 `cache_hit` requests and
1,582,245 physically prefetched tokens; M30-0001 reports the corresponding
reuse path as 53 `reuse_hit` requests and 1,574,774 new-prefill tokens.

Therefore the fixed replay is exact-trajectory evidence for the current
production reuse path, but it is not a discriminating before/after experiment
for the M30 implementation. The small 0.47% physical and 0.39% wall-time
differences cannot be attributed to the M30 changes. A meaningful attribution
experiment requires either a controlled `--no-session-reuse` production
baseline or an isolated GDN-only comparison with the existing GQA cache held
constant; neither is claimed here.

The code-coverage boundary is also explicit: the focused lifecycle test
instantiates `PrefillV2Model`, while `miinfer serve` routes through
`Qwen35RuntimeEngine`. The latter has its own pre-existing session-checkpoint
implementation that snapshots recurrent state/history and attention KV. The
canonical server run therefore validates the production checkpoint path and
its telemetry, but does not prove that the newer `PrefillV2Model` reusable
context is the path serving requests. Wiring those two paths together would be
a separate integration change and is intentionally not inferred from the
green focused test.
