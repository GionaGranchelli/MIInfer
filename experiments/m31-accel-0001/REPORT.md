# M31-ACCEL-0001 morning report

Deadline: 2026-10-12 10:00 Europe/Amsterdam
Status: IN PROGRESS. No campaign GPU run or optimization has been completed yet.

## Source identity and worktrees

- Starting source: `dc83f07f6adaaeec75a344b0155fa049553f2f7b`, fetched as the committed Z840 `m30/agent-workload-baseline` branch. It descends from `origin/main` at `3991b6bf40f810b63a100c46981e16964bb5434b`.
- Coordinator and worker branches/worktree SHAs are tracked in `STATUS.md`.

## 64K persistent-agent workload

Pending. Existing M30 evidence covers a 20-turn workload at approximately 5.1K cumulative prompt tokens with a 16K capacity. It does not qualify 64K persistent multi-turn execution. Existing single-prompt 64K execution is not a substitute.

## Decode and prefill

Pending fresh matched measurements. Audited historical data does not support a current promotion candidate: eager GQA decode is correctness-fixed with descriptive eager slowdown and neutral HIP Graph timing; MMQ/GQA variants have parity or performance failures. Existing persistent-prefix qualification fails token parity at token 11. No new performance change is authorized until a current bottleneck is measured.

## Memory and correctness

No campaign workload has been run yet. A past 64K call recorded about 29.45 GB model VRAM and a 102 C hotspot; repeat only under active telemetry and repository thermal guards.

## LC-0009 handoff

CPU ordinary-512 evidence was committed and pushed at `a5140f8`; its frozen visitor-use clause failed because output omitted visitor-use facts. Full Contract O QSUM decode coverage remains open. The 24-cell Lake inventory is preserved.

## Results and disposition

To be updated with exact commands, run IDs, hashes, telemetry, correctness outcomes, candidate SHAs, and explicit KEEP/REJECT/LEARN disposition. No performance gain is claimed.
