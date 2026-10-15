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

The first exact control used binary SHA-256
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
384 tokens and stopped at the output limit without a tool call.

The raw ratio against the earlier exact request is 6.16x faster for prefill,
despite 1,034 more tokens. This is a promising signal, not a qualified
speedup: prompt token counts differ, the binaries differ, and the exact
control's generated tool call was malformed.

## Agent behavior and task check

Repeated candidate continuations reused 66,514 of 66,517 prompt tokens, with
only three new prefill tokens. At output caps of 1,024 and 2,048, the model
did not produce a useful action. At a 4,096-token cap it named `replace_once`
but emitted `{}` arguments. The exact control also emitted `{}` for its
`read_file` call, so the malformed tool arguments do not isolate an attention
regression.

After the tool-call failure, the candidate returned the exact intended patch
in plain text: add `65536U` to the large-total loop in
`tests/gqa6_sparse_window_oracle.cpp`. I applied that one-line change and
built and ran `miinfer-gqa6-sparse-window-oracle` in the pinned ROCm 7.2.1
container; the command exited 0. This verifies the suggested edit and its
boundary oracle, but the model did not execute the edit through its tool call.

The test change is pushed in commit `1f92202` on
`perf/m31-gqa6-sparse-window-screen`.

## Same-binary control in progress

The old exact server completed before it was stopped; its logs and session
files were preserved. A fresh exact server now uses the candidate binary
`2b90a5ce898c205c60ef3f502bc1d08dd42bb1b5abb3b08bae3b597564437544` on GPU0,
with the sparse flag omitted. It receives the identical 66,517-token prompt
and output limit used by the cold candidate request. Its dedicated guard
monitors GPU0 by the verified bus mapping and stops at 85 C (warning at 80 C).

The same-binary exact run has not completed yet. Until it does, the 6.16x
ratio remains preliminary. The candidate also has not passed an autonomous
tool-executed coding task, persistent restore, or graph-replay semantic gate.

Raw logs and thermal telemetry are on Machinist under
`/home/machinist/m31-gqa6-longctx-results/EXP-0469-sparse-semantic/` and
`/home/machinist/m31-gqa6-longctx-results/EXP-0469-matched-exact/`.
