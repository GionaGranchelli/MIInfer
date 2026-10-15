# EXP-0469 — Sparse GQA6 64K coding-task screen

## Scope

Screen the opt-in sparse-window GQA6 Stage1 route on a real 65K-token MIInfer
repository prompt. Keep the exact default unchanged. GPU0 is the exact control
and GPU2 the candidate; Z840 is excluded.

The candidate attends to all keys through 2,048 tokens. Beyond that it attends
to the latest 2,048 keys and absolute anchors every 512 positions. The Q8 KV
cache and total context position remain intact, but attention contributions
from unselected keys are omitted. This is an approximate schedule.

## Cold full-model prefill

Both servers used the Qwen3.8-27B-Q4_K_M model and a 98,304-token context on
Machinist. The prompt was built from 755 MIInfer source/documentation files;
its context bytes SHA-256 was
`98755b381bffd786553916aa419cdd06a8dd008cfd8710787364640ffb0e7c74`.

An initial, non-matched screen used binary SHA-256
`9a9cbf89bac759e6ce8c18de3ca7282ff391015ea0ef54c8b75241c0e6e1b157` on GPU0
(BDF `0000:04:00.0`). It processed 65,483 prompt tokens with no prefix reuse:
prefill took 2,041.670 seconds (32.06 tok/s); first-token wall time was
2,041.931 seconds. It generated 69 tokens and emitted a `read_file` call with
empty arguments, so the harness could not continue the task.

The first sparse candidate used binary SHA-256
`2b90a5ce898c205c60ef3f502bc1d08dd42bb1b5abb3b08bae3b597564437544` on GPU2
(BDF `0000:88:00.0`). Route logs repeatedly showed
`miinfer_q8_gqa6_sparse_window_dispatch=1`. It processed 66,517 prompt tokens
with no prefix reuse: prefill took 331.595 seconds (200.60 tok/s), with
331.842-second wall time to the first visible delta. The request generated
384 tokens and stopped at the output limit without a tool call. This first
comparison is superseded by the matched pair below; its binaries and token
counts differed.

### Matched simultaneous pair

Both paths then used the same candidate binary
`2b90a5ce898c205c60ef3f502bc1d08dd42bb1b5abb3b08bae3b597564437544`, the same
model and exact 66,525-token prompt, fresh session directories, 384-token
output limit, and 98,304-token runtime capacity. They ran simultaneously on
GPU0 BDF `0000:04:00.0` (exact, sparse flag omitted) and GPU2 BDF
`0000:88:00.0` (sparse enabled). Both processed all prompt tokens, had zero
prefix reuse, and completed without runtime errors.

| Path | Prefill | Throughput | Whole request | Generated tokens | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Exact | 2,050.280 s | 32.45 tok/s | 2,076.810 s | 384 | Output limit; no content or tool call |
| Sparse | 331.315 s | 200.79 tok/s | 359.771 s | 384 | Output limit; no content or tool call |

This one simultaneous pair measured **6.19x faster prefill** for the sparse
route. The route trace confirmed sparse dispatch. The result is a matched
single-run measurement, not a repeated or general speed claim. Both GPUs were
configured with 180 W power caps. The exact route evaluates the full causal
KV range; at the final 66K position the sparse selector uses 2,174 of 66,525
keys (about 3.3%). Both cards showed 100% GPU utilization in monitoring
snapshots. Peak junction temperatures were 75 C on GPU0 and 68 C on GPU2,
below the 80 C warning threshold.

## Agent behavior and task check

Repeated candidate continuations of the first candidate prompt reused 66,514
of 66,517 prompt tokens, with only three new prefill tokens. At output caps of
1,024 and 2,048, the model did not produce a useful action. At a 4,096-token
cap it named `replace_once` but emitted `{}` arguments. The earlier exact
control also emitted `{}` for its `read_file` call, so the malformed tool
arguments do not isolate an attention regression.

For the matched pair, both follow-up requests reused 66,522 of 66,572 tokens
and prefetched the remaining 50. Exact prefill took 5.254 s and sparse took
4.139 s. Each generated 512 tokens and reached the output limit without
visible content or a tool call; total follow-up wall times were 40.77 s and
42.10 s, respectively.

### Process-restart restore

I restarted both candidate-binary servers with the preserved session files
and restored the 66,572-token state. The restore path reported
`persistent_session_restored=true` for both sessions, with 2,357,500,353
bytes of state, 66,569 reused prefix tokens, and three newly prefetched
tokens. Restore took 2.491 s on exact GPU0 and 2.348 s on sparse GPU2. The
first post-restore generation was empty at the 512-token cap, so I continued
each session from that same restored state with a request asking for the
specific boundary-oracle edit.

| Restored path | Follow-up TTFT | Follow-up wall | Generated tokens | Returned edit |
| --- | ---: | ---: | ---: | --- |
| Exact GPU0 | 6.117 s | 14.843 s | 131 | Correct replacement with 65,535, 65,536 and 65,537 boundaries |
| Sparse GPU2 | 4.334 s | 12.936 s | 121 | Incorrect duplicate 65,536 boundary; no tool call |

Both follow-ups reused the same restored 66,569-token prefix and prefetched
61 additional tokens. The exact response contained the intended patch in
plain text. The sparse response did not: it repeated `65536U` and omitted the
adjacent boundaries. Neither response executed a tool call. This proves the
session can be restored across server restart and retain long context, but
does not establish a correct autonomous coding-agent workflow on the sparse
path.

After the tool-call failure, the candidate returned the exact intended patch
in plain text: add `65536U` to the large-total loop in
`tests/gqa6_sparse_window_oracle.cpp`. I applied that one-line change and
built and ran `miinfer-gqa6-sparse-window-oracle` in the pinned ROCm 7.2.1
container; the command exited 0. This verifies the suggested edit and its
boundary oracle, but the model did not execute the edit through its tool call.

The test change is pushed in commit `1f92202` on
`perf/m31-gqa6-sparse-window-screen`.

The candidate did not execute the edit through its tool call. A correct
plain-text patch was returned by the exact path after explicit correction,
then applied and validated by the harness operator. This is evidence that the
exact path can recover the old task requirement and provide a valid edit, but
it is not an autonomous tool-executed coding session. Graph-replay semantics
were not tested in this 64K screen.

Raw logs and thermal telemetry are on Machinist under
`/home/machinist/m31-gqa6-longctx-results/EXP-0469-sparse-semantic/`,
`/home/machinist/m31-gqa6-longctx-results/EXP-0469-matched-exact/`, and
`/home/machinist/m31-gqa6-longctx-results/EXP-0469-sparse-repeat1/`. Restart
restore requests, server logs, and thermal telemetry are in
`/home/machinist/m31-gqa6-longctx-results/EXP-0469-restore-exact/` and
`/home/machinist/m31-gqa6-longctx-results/EXP-0469-restore-sparse/`.
