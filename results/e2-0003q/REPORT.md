# E2-0003Q qualification report

## Verdict

MMQ-only Gate/Up passes the applicable qualification gates on Z840: `QUALIFICATION_PASS_EXPERIMENTAL`. It produced the same 32 greedy tokens as control, reduced sampled peak VRAM by 7,248,957,440 bytes, and retained 89.8% of control decode throughput. Keep the route experimental; this does not authorize changing the production default. The A/B run count is one pair, and whole-model logit equivalence was not established.

## Source and build

- Benchmark source: `ed21f41318fd024c3223bff928bb8ac1cc2efcdd`; clean source archive SHA-256 `7486fb0ff3478b448895ac3319eb90348f87c9cd0f0514407171bd36a4c5b582`.
- Snapshot eager/graph-after-restore test source: `ab1d7161e2b18c1d77dd0930f77f3bb0efd8cba5`; source archive SHA-256 `449f83c4917cecf498719d82a0f75e042d7d136b98c1a203379c9552acedb9e9`.
- Pinned container: `localhost/miinfer-dev:rocm-7.2.1`, digest `sha256:bdc5ed42c985a6a333083e023225f248825a6e1511f38b6b50fbc7d5ace3fe9e`.
- Z840 GPU: BDF `0000:06:00.0`, unique ID `0x21678e17348c2f7`, one `gfx906` agent.
- Release build used HIP, tests, and benchmarks enabled; research tools disabled; target `gfx906`. The image does not ship `hipblasConfig.cmake`; the build used the minimal imported-target shim in `provenance/hipblasConfig.cmake` pointing at the image's installed hipBLAS header and library.
- The clean Release build succeeded. CTest: 35 total, 34 passed, 0 failed, 1 skipped (`package-archive-smoke`). The first CTest attempt could not find the unbuilt `physical-kv-view` executable; after building it, the final full run passed. The final snapshot test target was rebuilt from source commit `ab1d716` and passed its guarded runs below.

The clean base commit lacked the benchmark source and target present in the historical archive. That recovered benchmark also inherited the API default repetition penalty of 1.15; this qualification sets it explicitly to 1.0. The historical E2 and corrected E1/E3 outputs first diverge at token 7, after token 198 repeats. The repetition-penalty mismatch is the concrete sampler difference and a plausible cause of the changed greedy choice; the historical binary was not rerun to prove causality. Historical throughput and token results are not directly comparable. The recovery and hash audit are in `provenance/audit.md`.

## Runtime results

The benchmark used one binary, a deterministic 1,024-token prompt, 32 generated tokens, cold prefill, one measured iteration, and explicit sampler settings: temperature 0, top-p 1, top-k 1, repetition penalty 1, frequency/presence penalties 0, stop IDs disabled, repeat-last-n 256, seed 42, greedy argmax, reset enabled, HIP graph enabled. The only route difference was `MIINFER_EXPERIMENTAL_MMQ_GATEUP_ONLY=0` versus `1`.

| Metric | Control | MMQ-only | Result |
|---|---:|---:|---:|
| Prefill throughput | 245.382 tok/s | 246.984 tok/s | 100.7% of control |
| Decode throughput | 28.403 tok/s | 25.493 tok/s | 89.8% of control |
| Average decode-step latency | 35.207 ms | 39.227 ms | `decode_ms / 31` autoregressive steps |
| Sampled peak VRAM | 26,065,723,392 B | 18,816,765,952 B | 7,248,957,440 B saved |
| Model allocation request | 24,068,487,168 B | 16,938,170,368 B | 7,130,316,800 B less |
| Generated token IDs | 32 | 32 | Exact parity |
| Peak junction temperature | 63 C | 62 C | Below 80 C warning / 85 C stop |

The independent Gate/Up operation reference passed on the same BDF: cosine `0.999998`, relative L2 `0.001991`, max absolute error `0.000979885`, within the recorded thresholds. The graph diagnostic run captured and instantiated the decode graph, replayed it twice, and validated all 851 weight pointers across capture, instantiation, and replay. Default-off parsing was covered in CTest.

An opt-in `hipMalloc` interposer observed 972 successful calls during the candidate run; all occurred before `DECODE_BEGIN`, with zero calls from `DECODE_BEGIN` through `DECODE_END`. The source audit found no alternate HIP allocation API on the decode path. See `evidence/runtime-ed21f41/allocation-trace/` and `provenance/hip_malloc_trace.cpp`.

## Context lifecycle gates

All runs below used the Z840 model under the thermal guard and exited with cleanup complete:

- Exact-prefix reuse: passed zero-suffix repeats, suffix continuation, boundary lengths, mismatch fallback, and fresh-session parity.
- Snapshot/fork/rollback: passed snapshot restore, independent forks, nested rollback, shared/COW backing, invalid/released ID checks, eviction, reset, and cold replay parity.
- Graph after restore: passed inside the snapshot test. Graph capture/instantiation/replay and pointer validation succeeded after restoring a fork; first generated token matched the non-graph branch.
- Agent-branch workload: passed 10 operations with replay and rollback.
- COW branch workload: passed 8 operations and 8 snapshots.

All four lifecycle tests passed both with the experimental option unset and with `MIINFER_EXPERIMENTAL_MMQ_GATEUP_ONLY=1`. The final snapshot test compared the full two-token branch output from eager and graph decode, restored a fork, then replayed the already captured graph and matched the same output. Both default and MMQ-only runs passed with pointer validation enabled. The maximum sampled junction temperature across the guarded qualification logs was 71 C.

## Limits and skipped work

- The A/B result is one pair and one measured call per route; report medians are descriptive single samples. No second pair or prompt-2K run was made.
- Token parity is output-sequence parity only. Full logits/numerical equivalence was not checked.
- The allocation interposer observes the `hipMalloc` API used by this source; vendor-internal allocations through other APIs are outside its count. No full logits comparison was collected.
- No E2-0003Q workload was run on Machinist. The current host-access record says its 4 KiB H2D+D2H probe passed on 2026-10-10; that smoke probe alone does not qualify it for this Z840 result.
- No push or PR was created. The original checkout `/home/gionag/Development/mi50-m29-0006-common` remained untouched; work and evidence are in this isolated worktree.

## Required final fields

```text
E2-0003Q: PASS
SOURCE_REPRODUCIBLE: YES (clean committed source; fresh pinned Release build)
CLEAN_BUILD: PASS
BENCHMARK_SOURCE_COMMITTED: YES (ed21f41318fd024c3223bff928bb8ac1cc2efcdd)
EVIDENCE_ARCHIVE_COMPLETE: YES (all old and new SHA256SUMS entries verify)
SAMPLER_SETTINGS_VERIFIED: YES (explicit greedy settings; historical E2 used repetition penalty 1.15)
FULL_CTEST: 34 passed, 0 failed, 1 skipped
TARGETED_TESTS: experimental parser/default-off, graph/eager decode, quantized projection, allocation/destruction, and four guarded lifecycle tests PASS
GRAPH_CAPTURE: PASS
GRAPH_REPLAY: PASS (A/B diagnostic replays and replay after snapshot restore)
PERSISTENT_CONTEXT: PASS with experimental option unset and set
SNAPSHOT: PASS
FORK: PASS
ROLLBACK: PASS
MULTI_TURN: PASS
CONTROL_PREFILL: 245.382 tok/s
CANDIDATE_PREFILL: 246.984 tok/s
CONTROL_DECODE: 28.403 tok/s (35.207 ms/decode step)
CANDIDATE_DECODE: 25.493 tok/s (39.227 ms/decode step)
VRAM_SAVINGS: 7,248,957,440 bytes sampled peak
TOKEN_CORRECTNESS: 32/32 IDs match; no first divergence
NUMERICAL_REGRESSION: none confirmed; operation reference PASS; whole-model logits not compared
THERMAL_STATUS: PASS; maximum sampled junction 71 C, 80 C warning / 85 C stop
QUALIFICATION_VERDICT: QUALIFICATION_PASS_EXPERIMENTAL
OUTSTANDING_RISKS: one A/B pair; no prompt-2K; no full logits comparison; historical repetition-penalty cause is source-supported but not replayed with the old binary; no Machinist E2 workload
NEXT_SINGLE_GOAL: none for this qualification; keep MMQ-only experimental
BASE_SHA: e75dfcf8ececab5c032cb233f8aeed478268a1ba
FINAL_SHA: see final evidence commit
RAW_EVIDENCE: evidence/runtime-ed21f41/
COMMITS: a875b6c, 0bc6481, 7740032, 77dd2e3, ed21f41, 2fe062b, 56eb010, ab1d716, final evidence commit
```

## Evidence map

- Raw build, CTest, operation reference, graph diagnostic, A/B, and guarded lifecycle logs: `evidence/runtime-ed21f41/`.
- A/B machine-readable summary: `evidence/runtime-ed21f41/ab-1k-32/benchmark.json`.
- Exact tracked-source manifests: `provenance/source-manifest-ed21f41.sha256`, `provenance/source-manifest-2fe062b.sha256`, `provenance/source-manifest-56eb010.sha256`, and `provenance/source-manifest-ab1d716.sha256`.
- Runtime evidence archive SHA-256: `01e4ec186b5df891a45d90b6bf0d30df1960e2d47b0e5045b92f8c8f5dae145c`.
