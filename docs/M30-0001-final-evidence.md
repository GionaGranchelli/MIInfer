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

The 32 GiB Machinist 27B runtime attempt remains separately blocked by HIP out-of-memory during recurrent-layer construction and is not used as correctness evidence for the Z840 run.
