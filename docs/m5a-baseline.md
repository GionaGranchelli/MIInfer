# M5-A — Reproducible MI50 inference baseline

**Status:** COMPLETE for the current correctness-first C3 path  
**Date:** 2026-08-31  
**Build:** MI50 Release  
**Model:** Qwen3-8B Q4_0  

## Question

What is the reproducible end-to-end latency baseline for the closed M4-C3
text-facing MI50 path, with prompt ingestion and autoregressive decode reported
separately?

## Scope

This baseline measures the existing model-backed path without model loading or
GPU plan construction. Prompt ingestion is sequential batch-1 processing because
the current runtime has no batched prefill API. Decode measures the forward
calls after the first greedy token. No sampling, batching, graph capture, or
kernel optimization is included.

## Workload

```text
prompt text: hello
prompt token IDs: [14990]
requested generated tokens: 8
warm-up runs: 1
measured runs: 3
generation: greedy argmax
model SHA256: 458634762bea7dbe19f3ce0614465bafd15ee90e815229c427043afcf195d628
```

The measured sequence was identical on every run:

```text
8, 341, 286, 470, 330, 9707, 11, 330
```

All outputs were finite and all per-layer KV-cache lengths were validated.

## Environment

```text
GPU: gfx906 MI50-class device
total VRAM: 34342961152 bytes
model weights: 4768792576 bytes
planned weights: 4768792576 bytes
planned workspace: 63488 bytes
```

The complete machine state is retained by the benchmark runner in the result
directory, including before/after environment captures and active `rocm-smi`
telemetry. The benchmark result was captured at:

```text
bench/results/20260831T191803Z-326236/
```

That first capture was made while the M5-A benchmark changes were uncommitted;
the canonical rerun after the M5-A commit should be used for future comparisons.

## Result

| Metric | Mean | Median | Throughput |
|---|---:|---:|---:|
| Sequential prompt ingestion | 37.376 ms | 37.415 ms | 26.755 tok/s |
| TTFT, including reset | 37.860 ms | 37.910 ms | — |
| Post-first-token decode (7 forward calls) | 298.415 ms | see JSON | 23.457 tok/s |

The raw per-run values, min/max, standard deviation, and exact median are in
`result.json`; no timing samples were discarded.

## Interpretation

This is the first end-to-end M5-A baseline for MIInfer. It is not a claim about
optimized-runtime performance: the current decode API deliberately copies
diagnostic traces and allocates per-token temporary buffers. Future M5 changes
must retain this correctness workload, record comparable hardware/software
state, and report performance and memory tradeoffs against this baseline.

## Reproduction

```bash
cmake --preset mi50-release
cmake --build --preset mi50-release --target miinfer-qwen3-inference-bench
scripts/run-m5a-baseline.sh /path/to/Qwen3-8B-q4_0-b968826d.gguf
```

The script delegates to `scripts/run-bench.sh`, which creates a fresh result
directory under `bench/results/`.

## Decision

`KEEP` — the benchmark is the M5-A baseline control. No performance change was
made to the inference implementation.
