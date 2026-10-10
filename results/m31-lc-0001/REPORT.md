# M31-LC-0001 — MMQ-Only Long-Context Advantage

## Decision

`CORRECTNESS_BLOCKED`

At the first required gate (2K prompt), fused and MMQ-only each returned 16 valid token IDs, but their greedy sequences reproducibly diverged at generated token 5: fused `271`, MMQ-only `74455`. The divergence repeated in two same-source A/B pairs. The E2 selection gate requires exact route token parity, so Gate A did not pass and the sequential 4K/8K gates were not run. This is a numerical/output divergence, not a capacity or thermal abort.

The isolated Gate/Up operation reference from the reused E2 evidence passed its existing tolerance (cosine `0.999998`, relative L2 `0.001991`, max absolute error `0.000979885`; thresholds `0.995` / `0.02`). That component result does not establish full-model correctness for the 2K sequence. The MMQ route therefore remains experimental and this milestone does not confirm a long-context advantage.

## Final fields

```text
M31-LC-0001: BLOCKED
SOURCE_SHA: cb99829ee662aa745c327496bd922c4ffcea5792 (runtime benchmark and harness source)
REMOTE_SHA: see final evidence commit; verified against origin branch
HOST: Z840 / fedora; MI50 gfx906; BDF 0000:06:00.0; unique ID 0x21678e17348c2f7
MODEL_SHA: 7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169

MAX_TESTED_CONTEXT: 2K prompt (2,048 tokens; repeated twice)
MAX_QUALIFIED_CONTEXT: 1K from the prior E2-0003Q gate; M31-LC Gate A did not qualify 2K
ESTIMATED_128K_MEMORY_BUDGET: MMQ-only 27.046 GB peak estimate plus 3.434 GB reserve leaves 3.862 GB; fused 34.177 GB estimate is unsafe against the 34.343 GB device total and reserve

CONTEXT_RESULTS:
  1K historical E2-0003Q: exact 32/32 token parity; control/MMQ-only sampled peak 26,065,723,392 / 18,816,765,952 B
  2K: exact 16/16 valid tokens per route, but cross-route parity FAIL at token 5; sampled peak 26,132,242,432 / 18,883,702,784 B in the instrumented pair
  4K: NOT_RUN (2K exact-parity gate failed)
  8K: NOT_RUN (2K exact-parity gate failed; primary target unresolved)
  32K: NOT_RUN (conditional gate not reached)
  16K/64K/128K: estimates only; no workloads run
PREFILL_SCALING: at 2K, control 241.457 tok/s; MMQ-only 242.398 tok/s (one instrumented pair; earlier 2K pair was 241.876 / 242.223)
DECODE_SCALING: at 2K, control 28.271 tok/s; MMQ-only 25.377 tok/s (89.8%); earlier pair was 28.277 / 25.383
TTFT_SCALING: at 2K, control 8,481.839 ms; MMQ-only 8,448.927 ms; 1K historical TTFT was not recorded
VRAM_SCALING: at 2K, control 26,132,242,432 B; MMQ-only 18,883,702,784 B; 7,248,539,648 B saved in the instrumented pair (earlier pair saved 7,248,809,984 B)

GRAPH_REPLAY: HIP graph reported active on both 2K routes; separate 2K eager/graph equivalence not run
PREFIX_REUSE: NOT_RUN in this milestone; prior integrated M31-0002 parity failure remains open
SNAPSHOT_FORK_ROLLBACK: NOT_RUN in this milestone
CORRECTNESS: BLOCKED by repeated fused/MMQ greedy output divergence at token 5; isolated operation reference passes but full-model logits were not compared
THERMAL_STATUS: PASS; peak junction 68 C; no warning/abort; cleanup completed on all four guarded calls

FUSED_VS_MMQ: memory saving confirmed at 2K; prefill preserved; decode retained 89.8%; correctness gate failed
LONG_CONTEXT_ADVANTAGE: NOT_CONFIRMED
LIMITING_FACTOR: correctness/output parity, not available VRAM, thermal behavior, or the measured decode threshold

TESTS: 8 local A/B harness tests PASS; six focused guarded CTest smoke tests PASS; reused E2 full CTest 34 passed, 0 failed, 1 skipped
SKIPPED: 4K/8K/32K workload gates and persistent-agent phase stopped after reproducible Gate A parity failure; no 64K/128K workload; no full-model logit comparison
RAW_EVIDENCE: results/m31-lc-0001/; prior qualified baseline and operation reference: results/e2-0003q/evidence/runtime-ed21f41/
COMMITS: 8a8591a base; 667f3e7 harness; a49b185 preflight/memory envelope; cb99829 TTFT instrumentation; final evidence commit
PUSH_VERIFIED: yes; final local and remote qualification-branch SHAs match

NEXT_SINGLE_ENGINEERING_GOAL: resolve or explicitly accept the repeatable 2K fused/MMQ token divergence against the existing correctness contract before advancing the context ladder; do not add another optimization
```

## Phase 0 — preflight

Z840 SSH succeeded within two attempts. The live host, GPU identity, idle state, temperature, model SHA, and pinned image digest matched the qualification record. Six focused smoke tests passed under the real 80 C warning / 85 C stop guard. The M31 guard self-test also passed. Full E2-0003Q clean build and CTest evidence was reused rather than repeating its 1K A/B qualification.

## Phase 1 — memory estimate

See [phase1-memory-envelope.md](phase1-memory-envelope.md) for measured allocation inputs, source-derived workspace rules, 2K–128K estimates, and the 10% safety reserve. It estimates MMQ-only memory budget through 128K, but that is not a 128K support claim.

## Phase 2 — guarded scaling

Two 2K fused/MMQ pairs used the same model, deterministic prompt (seed 77), one binary per pair, temperature 0, top-p 1, top-k 1, repetition penalty 1, frequency/presence penalties 0, no stop IDs, repeat-last-n 256, seed 42, and HIP graph enabled. Only `MIINFER_EXPERIMENTAL_MMQ_GATEUP_ONLY` changed within each pair. All four calls completed with clean process-group teardown and no thermal abort.

The first pair used the existing benchmark binary before the TTFT output field was added; the second pair used the instrumented binary. Outputs reproduced exactly per route across pairs, including the fused/MMQ divergence. Both pairs showed approximately `7.249 GB` sampled peak-VRAM savings and approximately `89.8%` decode throughput retention. Since exact token parity is an established selection gate, Gate A failed and the ladder stopped before 4K.

## Phase 3–5 — not advanced

No persistent-agent workload was run after Gate A failed; the highest previously qualified context remains the E2 1K result. This task does not resolve the separate M31 64K/TG128 or integrated prefix-reuse failures. No throughput is extrapolated to untested lengths.

## Evidence map

- 2K first A/B pair: `2k/benchmark.json` and route records/logs.
- Repeated 2K pair with TTFT: `2k-ttft/benchmark.json` and route records/logs.
- Live smoke and thermal guard: `phase0-smoke-final.ctest.log`, `phase0-smoke-final.guard.log`.
- Build and source identity: `ttft-build.log`, `source-manifest.json`.
- Memory estimate: `phase1-memory-envelope.md`.
