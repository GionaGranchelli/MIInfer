# M31-0003 — Canonical Agent Workload

## Scope

Run a deterministic 20–50 turn coding-agent session over the frozen M31
runtime. The workload is a regression artifact, not an open-ended model
conversation and not a rerun of the M30 86-request campaign.

## Required sequence

```text
conversation -> code generation -> tool invocation -> tool result
-> continuation -> branch -> failed attempt -> rollback
-> alternate branch -> long tool result -> continuation
-> snapshot -> another branch
```

The same transcript, repository excerpts, tool results, model settings, and
source manifest must be used for all paths:

1. cold/replay baseline;
2. persistent-context path;
3. persistent-context plus COW path.

## Headline metrics

```text
agent_runtime_speedup = cold_replay_wall_ms / persistent_runtime_wall_ms
replay_avoided = 1 - replayed_tokens / baseline_replayed_tokens
```

All paths must pass deterministic output/parity gates. The result must retain
request-level timing, prompt/replayed/avoided tokens, branch events, and the
exact workload/source manifest.

## Evidence

```text
M31_BASELINE_SHA=99840d10eabe5e64fe2564d34d3d3fcd42b43521
M31_0003_WORKLOAD_SHA=7ee03f8678fd59c82c88246e5b5250464529c216
M31_0003_RESULT=FAIL
```

The canonical native workload is implemented in
`bench/m31_0003_canonical_agent_workload.cpp`; the Python runner remains a
dry-run/API manifest helper. The card-1 native run completed 20 deterministic
turns with one branch/rollback sequence. Its artifact is
`results-m31-0003-card1-native.txt` (SHA-256
`640d1bec3d0e3957ae7d0acd6ecbce08c883a236d264124b0a0af9d429775765`). It
measured 256681.460 ms cold/replay wall time, 47176.381 ms persistent wall
time, 5.441x speedup, 99.5% replay avoidance, 14 COW events, and
2928672768 COW bytes copied. The output-parity gate failed, so these are
diagnostic measurements and are not promoted as a qualification pass.
