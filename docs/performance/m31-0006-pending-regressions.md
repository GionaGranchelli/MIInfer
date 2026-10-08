# M31-0006 — GPU regression status

Status: targeted M31-0004 and M31-0005 regressions passed on the Z840 MI50
under the real-sensor workload guard. All GPU runs were sequential.

## M31-0004

- Exact-prefix test passed cold/warm reuse, repeated zero-suffix hits, suffix
  continuation, 511/512/513 and 1024/1025 prefix boundaries, mismatch
  fallback, and fresh-session output parity.
- Snapshot/fork/rollback test passed snapshot mutation, fork, nested branch,
  rollback, invalid snapshot, release, shared state, COW, eviction, reset, and
  cold parity.
- Agent-branch workload passed 10 branches with output parity; 103 physical
  replay tokens in the cold path vs 23 in the reused path (80 avoided).

The first exact-prefix attempt OOMed because the test kept a fresh model
instance alive while constructing a larger boundary instance. The test now
destroys the fresh instance before the second construction; the retry passed.
This was a test-lifetime resource bug, not a production correctness failure.

## M31-0005

The direct eager regression validates the poisoned future KV boundary. The
expanded test also checks eager-vs-HIP-Graph output parity and poisoned future
slots at positions 1, 511, and 512. All pass at `1e-3` absolute tolerance,
including the 512-token page transition.

The first expanded assertion used a zero gate while expecting output `1.0`;
`sigmoid(0)` correctly scales it to `0.5`. The fixture now uses a saturating
positive gate to isolate KV causality. No kernel change was required.

## Raw evidence

- [Exact-prefix pass](../../results/m31-0006-gpu-regressions/m31-0004-exact-prefix-pass.log)
  — SHA-256 `f4d2ccbf6cbc76e107119907057702e69413804b07b28c60f4d25713f1fe3cbd`
- [Snapshot/fork/rollback pass](../../results/m31-0006-gpu-regressions/m31-0004-snapshot-fork-rollback-pass.log)
  — SHA-256 `4d9d98cc01988eae4ba35416e4bb082ae79acccad05bdabed9821a2fe82025cb`
- [Agent branch pass](../../results/m31-0006-gpu-regressions/m31-0004-agent-branch-workload-pass.log)
  — SHA-256 `87ab7fd2d68150bedbf57dd623493d941ff2f5a08aee2d62073c0b64a1627e0e`
- [Eager/graph page-boundary pass](../../results/m31-0006-gpu-regressions/m31-0005-graph-eager-page-boundary-pass.log), SHA-256 `ff61510d7fd8d7aee196d87d79b18f67a59398d17feec4f135971d6a6778b73d`
- [Original eager boundary pass](../../results/m31-0006-gpu-regressions/m31-0005-boundary-pass.log)
  — SHA-256 `9ec8bce1a61e55bc04a3cb7add466bc8dba79772c363491808090fa915e94d3`
- [Production graph decode smoke](../../results/m31-0006-production-graph-decode-smoke.log)
  — SHA-256 `17b4ce26c88332dfe8cd247be90e2f9963e0c8212aacf02f0779a93bae4b1690`

The short production smoke used an 8-token prompt, generated four tokens with
HIP Graph enabled, and completed with finite timing. It is integration evidence,
not long-context correctness or an end-to-end performance comparison. No full
128K model prefill was run.
