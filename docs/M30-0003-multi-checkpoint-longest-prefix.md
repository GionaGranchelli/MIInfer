# M30-0003 Multi-Checkpoint and Longest-Prefix Reuse

Status: bounded alternate-path hypothesis passes; production serving
implementation is not yet claimed.

## Verified current state

The serving authority is still `cmd_serve -> PrefillV2Model`.
`ReusableContext` retains one in-memory GDN checkpoint, its token vector, and
an optional final hidden vector. The active `PrefillV2Model` owns the 16 GQA
KV caches separately.

The repository already contains two non-equivalent multi-checkpoint paths:

1. `PersistentSession::find_matching_session` scans multiple `.miinfer`
   files and selects the longest exact token-prefix match. A loaded session
   contains both GDN and GQA state, but it is disk/host-backed and does not
   contain the final hidden payload needed for a zero-suffix hit.
2. `Qwen35RuntimeEngine` retains up to eight full GPU snapshots under a 3 GiB
   budget and uses a radix index for longest-prefix selection. This is an
   alternate compatibility/runtime path, not `miinfer serve` authority.

The server already accepts `--session-dir` and passes it as
`persistent_session_dir`; this provides bounded multi-file retention when
callers explicitly opt into it. Its in-memory telemetry still reports the
single `ReusableContext` checkpoint as count one.

## Feasibility boundary

Adding multiple copies of only `ReusableContext` would be incorrect. A branch
restore must restore GDN state and the matching GQA KV prefix together; the
currently active KV caches cannot be reused after restoring an older GDN
checkpoint. Full in-memory snapshots therefore consume KV-sized VRAM, which
is incompatible with the observed 128K serving envelope unless a measured
budget and eviction policy are introduced.

The smallest safe next experiment is therefore a causal branch/retry workload
using the existing disk session path, with several retained checkpoint files,
longest-prefix selection, suffix token counts, restore cost, and output parity
against cold replay. COW, fork, rollback, tail replay, and a new serving
authority remain out of scope.

## Evidence boundary

The exact local `HEAD` was transferred to an isolated Machinist checkout and
built successfully with the target's ROCm HIP toolchain. With the required
layer-major runtime configuration, the bounded length checks passed for 512,
640, 3991, and 8192 token prompts. The explicit branch-only check then passed:

```text
session_branch_check=PASS older_prefix_reuse=2048 return_branch_reuse=3584
cached_entries=5 cached_bytes=2203582464
```

The divergent branch reused the older compatible checkpoint, the return branch
reused its longer checkpoint, and both generated-token sequences matched cold
replay. This proves the bounded longest-prefix mechanism in
`Qwen35RuntimeEngine`, not in `miinfer serve`.

The exact local build was also exercised through `miinfer serve` with
`--session-reuse --session-dir` on the Machinist. Exact-repeat reuse passed:
2808 prompt tokens reused, zero replayed, and approximately 0.75 ms restore.
Three `.miinfer` files were written for a base/A/B sequence, but the divergent
branch did not reuse the base file. Matcher tracing showed the saved terminal
prompt at 3009 tokens and the branch at 3010 tokens with `match=0`; the
ChatML terminal assistant marker means saving only the complete request does
not create an intermediate prefix checkpoint for the next branch.

Therefore persistent session selection is a storage primitive, not yet the
serving-side M30-0003 solution. The serving implementation still needs to
capture compatible intermediate GDN+GQA state together during prefill (or
another equivalent full-state mechanism) before it can claim branch replay
reduction.

An older dirty Machinist binary was also probed before the isolated build and
failed its first reuse assertion (`length=512`, `retained=0`); that result is
rejected because it did not match the current source/configuration contract.

No new full 86-request canonical run is justified before the targeted branch
workload exists. The M30-0002 residual evidence remains the baseline: 33
cold/fallback requests and 1,516,386 replayed prefix tokens.

## Exit criteria for implementation

- retain at least two compatible checkpoints in the serving-authority path;
- select the longest exact matching prefix;
- restore matching GDN and GQA state together;
- execute only the unmatched suffix;
- prove token/output parity with cold replay and report checkpoint count/bytes;
- measure branch/retry replay reduction and disk/restore overhead.
