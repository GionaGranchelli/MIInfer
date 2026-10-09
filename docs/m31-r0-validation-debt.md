# M31-R0 Validation-Debt Register

**2026-10-09 — Provisional inventory.** This report is based on milestone status summaries supplied by the maintainer and the connected GitHub remote. It is **not** a substitute for inspecting the Z840/Machinist worktrees and their raw result directories.

Connected remote `main`: `3991b6bf40f810b63a100c46981e16964bb5434b` (last visible October 2). Local M31 commits reported by maintainer are not accessible on that remote. **Do not reset main or attempt to infer missing SHAs from GitHub.**

Known released v0.2.0 qualified source: `94fad71ee19f539ce2ec0c7e100ad97d031dbefa`; its qualification applies to the specific release artifact/workloads, not future M31 behavior.

| Work | Reported commit | Reported state | Missing verification | Proposed disposition | Accountable |
| --- | --- | --- | --- | --- | --- |
| M31-0003 GDN lazy checkpoint allocation | `849b14b` | Host-only tests; GPU pending | device memory delta, prefix reuse | QUALIFY when local source available; do not label GPU_VALIDATED | Repo owner / Codex |
| M31-0004 cache residency and hash update | `b9533bc`, `43f8f427...` | Host-only tests; GPU regression unrun | cache boundaries, snapshot mutation, output parity | QUALIFY; correctness fix requires special handling | Repo owner / Codex |
| M31-0005 eager KV-boundary correction | `54d2fbed6e1b5f8dd8ff7c7293bff5144fe09fc7` | Host-only tests; GPU pending | poisoned future KV slot, eager/graph path parity | QUALIFY; avoid default-path regression | Repo owner / Codex |
| M31-0002T reduced harness | uncommitted on host at last report | 1K guarded diagnostic, token 7 divergent | numerical equivalence, E3 benchmark contract | ISOLATE instrumentation until validated | Repo owner / Codex |
| ROCm 6.2 linker libxml2 dependency | uncommitted/environment | full HIP compilation blocked in prior reports | reproducible build resolution | BLOCKED infra issue, not optimization | Repo owner / Codex |

## Required local audit

For each item, inspect actual source/working tree on both hosts, confirm commit relationships, diff exposed production path, list applicable tests and risks, and choose **one** final disposition:

- QUALIFY: make GPU correctness/performance evidence with targeted tests.
- ISOLATE: keep out of default execution via an isolated branch or explicit default-off gate; prove gate behavior.
- REVERT: justified removal after reviewing correctness/dependencies, never silently discard.
- RETAIN_AS_EXPERIMENT: document candidate branch and acceptance evidence needed.
- BLOCKED: record precise external condition and cheapest unblock action.

**Critical nuance:** Do not automatically isolate a correctness fix in a way that reactivates the known defect; its behavior and tests must drive the disposition. Do not mark an optimization as proven merely because compilation or CPU tests pass.

## Checkpoint

Link each final disposition to [M31-R0 #9](https://github.com/GionaGranchelli/MIInfer/issues/9). After [E1 #6](https://github.com/GionaGranchelli/MIInfer/issues/6), [E2 #7](https://github.com/GionaGranchelli/MIInfer/issues/7), and [E3 #8](https://github.com/GionaGranchelli/MIInfer/issues/8) close or explicitly block, select exactly one next optimization.
