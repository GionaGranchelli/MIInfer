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
M31_0003_WORKLOAD_SHA=<filled after implementation>
M31_0003_RESULT=<PASS or FAIL>
```
