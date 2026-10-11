# M31-ACCEL-0001 live status

Updated: 2026-10-12 00:25 Europe/Amsterdam
Deadline: 2026-10-12 10:00 Europe/Amsterdam

## Source and ownership

- Verified source anchor: `dc83f07f6adaaeec75a344b0155fa049553f2f7b` (`m30/agent-workload-baseline` on Z840); its merge base with `origin/main` is `3991b6bf40f810b63a100c46981e16964bb5434b`.
- Coordinator: branch `experiment/m31-accel-0001-coordinator`, worktree `/tmp/m31-accel-0001-coordinator`, start SHA `dc83f07f6adaaeec75a344b0155fa049553f2f7b`.
- Agent 64K: branch `experiment/m31-accel-0001-agent64k`, worktree `/tmp/m31-accel-0001-agent64k`, same start SHA; implementing only a bounded persistent multi-turn workload. No GPU runs assigned.
- Z840 `/home/fedora-workstation/Development/mi50` remains dirty (tracked M30 workload/result edits plus untracked artifacts/core); do not modify, clean, build, or run from it.
- Machinist `/home/machinist/Development/MIInfer` remains dirty at `d8b833637f85537d6205c9480129d95c819e6d61`; do not modify or build in place.

## Host/resource check (2026-10-12 00:21 local)

- Z840 SSH succeeded. MI50 idle, 29 C junction, 0% use, about 10.9 MB sampled VRAM used. Lake is mounted read/write at `/mnt/lc9-lake`, 1.6 TB free. No active MIInfer/M31 process or tmux session found.
- Z840 campaign source is available only in the dirty M30 checkout. Work must use a separate worktree.
- Machinist SSH succeeded; no active MIInfer/M31 process or tmux session found. Its checkout is dirty; CPU work must use an isolated worktree if needed. Do not use its MI50 for qualification due known H2D issue.

## Current evidence / primary gate

- Highest priority is exactly 65,536 prompt/context tokens followed by at least eight sequential persistent turns with reuse, snapshots where supported, position/state continuity, bounded outputs, and retained per-turn metrics.
- The latest audited native M30 run is 20 turns and passes replay/parity checks at roughly 5.1K cumulative prompt tokens, but capacity is hardcoded to 16K. This is not 64K evidence.
- A separate 64K run is single-prompt/warmed execution and lacks persistent multi-turn reuse; a previous such call reported a 102 C hotspot. Check thermal guards before any new MI50 job.
- Prefix reuse qualification at 8K+512 in M31-0002 diverged at generated token 11 (expected 3299, actual 4434). Therefore persistent suffix performance is unqualified until correctness is established.
- Protocol performance frontier says no active performance PRIMARY after V2-0044 closeout; no performance-sensitive code change is authorized before refreshed attribution. This campaign's user-specified 64K reliability goal is the active campaign priority.

## Queue and next actions

1. Agent returns the 64K workload branch; review the exact diff, build on isolated source, and inspect memory/capacity sizing.
2. Prepare an isolated Z840 worktree at the reviewed candidate SHA; serialize one guarded GPU workload at a time.
3. First attempt: one 64K multi-turn operational/correctness run with retained logs and telemetry. Stop at configured thermal limits. Do not begin an unbounded matrix.
4. Decode/prefill optimization only after a current bottleneck is measured and correctness contract is pinned; all current reported decode/prefill items are historical or parity-limited.
5. LC-0009 stays bounded. Its CPU ordinary-512 result is pushed on `experiment/m31-lc-0009-quantized-correctness` at `a5140f8`; QSUM decode / broader Contract O coverage remain open. Lake inventory hashes are preserved.

## Checkpoint discipline

One GPU job at a time on Z840. Preserve all dirty worktrees and existing logs. Record each run's PID, exact command, source/model/image identity, input IDs, exit status, hashes, temperatures, VRAM and correctness result before starting another.
