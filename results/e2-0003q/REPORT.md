# E2-0003Q qualification report

## Verdict

MMQ-only Gate/Up passes the requested **single-pair selection gates** on Z840. It produced the same 32 greedy tokens as control, reduced sampled peak VRAM by 7,248,957,440 bytes, and retained 89.8% of control decode throughput. This supports further evaluation with the option still opt-in; it is not evidence for changing the default. The run count is one pair, and whole-model numerical equivalence was not established.

## Source and build

- Benchmark source: `ed21f41318fd024c3223bff928bb8ac1cc2efcdd`; clean source archive SHA-256 `7486fb0ff3478b448895ac3319eb90348f87c9cd0f0514407171bd36a4c5b582`.
- Snapshot graph-after-restore test source: `2fe062b55f831a532cb2cf744ca7068258f72345`; source archive SHA-256 `fb9694e984f123d479dabbd19718e7d7954455a321beb79fb45d342d37a240f9`.
- Pinned container: `localhost/miinfer-dev:rocm-7.2.1`, digest `sha256:bdc5ed42c985a6a333083e023225f248825a6e1511f38b6b50fbc7d5ace3fe9e`.
- Z840 GPU: BDF `0000:06:00.0`, unique ID `0x21678e17348c2f7`, one `gfx906` agent.
- Release build used HIP, tests, and benchmarks enabled; research tools disabled; target `gfx906`. The image does not ship `hipblasConfig.cmake`; the build used the minimal imported-target shim in `provenance/hipblasConfig.cmake` pointing at the image's installed hipBLAS header and library.
- The clean Release build succeeded. CTest: 35 total, 34 passed, 0 failed, 1 skipped (`package-archive-smoke`). The updated snapshot test target was rebuilt from source commit `2fe062b` and passed its guarded run below.

The clean base commit lacked the benchmark source and target present in the historical archive. That recovered benchmark also inherited the API default repetition penalty of 1.15; this qualification sets it explicitly to 1.0. Historical throughput and token results are not directly comparable. The recovery and hash audit are in `provenance/audit.md`.

## Runtime results

The benchmark used one binary, a deterministic 1,024-token prompt, 32 generated tokens, cold prefill, one measured iteration, and explicit sampler settings: temperature 0, top-p 1, top-k 1, repetition penalty 1, frequency/presence penalties 0, stop IDs disabled, repeat-last-n 256, seed 42, greedy argmax, reset enabled, HIP graph enabled. The only route difference was `MIINFER_EXPERIMENTAL_MMQ_GATEUP_ONLY=0` versus `1`.

| Metric | Control | MMQ-only | Result |
|---|---:|---:|---:|
| Prefill throughput | 245.382 tok/s | 246.984 tok/s | 100.7% of control |
| Decode throughput | 28.403 tok/s | 25.493 tok/s | 89.8% of control |
| Sampled peak VRAM | 26,065,723,392 B | 18,816,765,952 B | 7,248,957,440 B saved |
| Model allocation request | 24,068,487,168 B | 16,938,170,368 B | 7,130,316,800 B less |
| Generated token IDs | 32 | 32 | Exact parity |
| Peak junction temperature | 63 C | 62 C | Below 80 C warning / 85 C stop |

The independent Gate/Up operation reference passed on the same BDF: cosine `0.999998`, relative L2 `0.001991`, max absolute error `0.000979885`, within the recorded thresholds. The graph diagnostic run captured and instantiated the decode graph, replayed it twice, and validated all 851 weight pointers across capture, instantiation, and replay. Default-off parsing was covered in CTest. Source inspection found no `hipMalloc` in the generated-token decode loop; this is a source audit, not a runtime allocation trace.

## Context lifecycle gates

All runs below used the Z840 model under the thermal guard and exited with cleanup complete:

- Exact-prefix reuse: passed zero-suffix repeats, suffix continuation, boundary lengths, mismatch fallback, and fresh-session parity.
- Snapshot/fork/rollback: passed snapshot restore, independent forks, nested rollback, shared/COW backing, invalid/released ID checks, eviction, reset, and cold replay parity.
- Graph after restore: passed inside the snapshot test. Graph capture/instantiation/replay and pointer validation succeeded after restoring a fork; first generated token matched the non-graph branch.
- Agent-branch workload: passed 10 operations with replay and rollback.
- COW branch workload: passed 8 operations and 8 snapshots.

## Limits and skipped work

- The A/B result is one pair and one measured call per route; report medians are descriptive single samples. No second pair or prompt-2K run was made.
- Token parity is output-sequence parity only. Full logits/numerical equivalence was not checked.
- The allocation finding is source inspection plus sampled VRAM; no per-token runtime allocator trace was collected.
- No Machinist run was made. Its prior qualification record identifies a failing 4 KiB host-to-device HIP copy prerequisite; that machine was not requalified here.
- No push or PR was created. The original checkout `/home/gionag/Development/mi50-m29-0006-common` remained untouched; work and evidence are in this isolated worktree.

## Evidence map

- Raw build, CTest, operation reference, graph diagnostic, A/B, and guarded lifecycle logs: `evidence/runtime-ed21f41/`.
- A/B machine-readable summary: `evidence/runtime-ed21f41/ab-1k-32/benchmark.json`.
- Exact tracked-source manifests: `provenance/source-manifest-ed21f41.sha256` and `provenance/source-manifest-2fe062b.sha256`.
- Runtime evidence archive SHA-256: `7a34a45345954f1d65a6e42b3a2e08116ff3d56f095725129f7ec7001161265f`.
