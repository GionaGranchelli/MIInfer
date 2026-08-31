# Benchmarks

The repository currently contains one benchmark: `miinfer-bench`, a trivial
HIP vector-add operation. It validates device execution, HIP-event timing, and
machine-readable result handling; it is not an MIInfer inference-performance
benchmark.

Build and run it on the target machine:

```bash
cmake --preset mi50-release
cmake --build --preset mi50-release
./build/mi50-release/miinfer-bench --warmup 5 --iterations 100
```

The benchmark selects the first gfx906 device by default. Use
`--device INDEX` to select one explicitly. Selecting a non-gfx906 device is a
hard error. Timing uses HIP events around the asynchronous kernel launch and
includes kernel execution, not host wall-clock launch overhead.

Warm-up launches are excluded from statistics. The measured iterations report
mean, median, minimum, maximum, and standard deviation in microseconds. The
JSON includes every measured HIP-event sample in `samples_us`; no samples are
discarded by the harness. The output is JSON on standard output;
`--json-output PATH` writes the same JSON to a file while human-readable status
is written to standard error. Successful results also include the compiled
MIInfer commit and dirty-state marker.

The runner captures the environment before and after a benchmark:

```bash
scripts/run-bench.sh ./build/mi50-release/miinfer-bench \
  --warmup 5 --iterations 100
```

Results are stored under `bench/results/<run-id>/` as `environment-before.json`,
`result.json`, `telemetry.jsonl`, and `environment-after.json`. The runner
samples `rocm-smi` during the benchmark at 250 ms by default; telemetry
timestamps are UTC with millisecond precision. Set
`MIINFER_TELEMETRY_INTERVAL_MS` to change that interval or
`MIINFER_TELEMETRY=0` to disable it. Each telemetry line is a complete JSON
object, so clock and thermal drift can be inspected without a special parser.

The environment script records `UNAVAILABLE` when a command or metric is not
exposed by the local ROCm/Linux installation. The initial benchmark must not be
used to claim inference performance or to substitute for representative
model-shape experiments.

## End-to-end M5-A baseline

`miinfer-qwen3-inference-bench` measures the current model-backed MI50 path
without including model loading or plan construction. It reports reset time,
sequential batch-1 prompt ingestion (the current implementation has no batched
prefill API), time to first token, and decode forward time after the first
greedy token. The benchmark keeps raw per-run samples in its JSON result and
requires every measured run to produce the same finite token sequence.

The canonical short baseline uses the closed C3 `hello` workload:

```bash
scripts/run-bench.sh ./build/mi50-release/miinfer-qwen3-inference-bench \
  /path/to/Qwen3-8B-q4_0-b968826d.gguf \
  --prompt hello --generated-tokens 8 --warmup 1 --iterations 3
```

Use `--prompt-repeat N` to measure a longer sequential prompt made from the
same encoded prompt, or `--prompt-ids CSV` for a fully explicit token workload.
The output records the model SHA256, build metadata, planned weight/workspace
bytes, total device VRAM, timing definitions, generated IDs, and all timing
samples. The runner additionally captures before/after environment state and
active GPU telemetry; peak VRAM and clocks must be read from those artifacts.

This is a baseline for the correctness-first C3 implementation. It includes
the current diagnostic trace copies and per-token allocations in the decode
API, so it is not yet an optimized-runtime performance claim.

## M5-B steady-state decode profile

`miinfer-qwen3-decode-profile` profiles one warmed position-1 decode for the
fixed token pair `14990` at position 0 followed by `8` at position 1. It uses
opt-in HIP events around operation launches and synchronous timing around
device copies. The profile reports operation-family GPU time, copy time, and
dispatch counts, plus an end-to-end wall-clock value. Because profiling
synchronizes each scoped operation and the correctness-first decode API copies
diagnostic traces, its wall time is not comparable to the M5-A throughput
baseline.

Run the reproducible profile through the environment/telemetry runner:

```bash
cmake --preset mi50-release
cmake --build --preset mi50-release --target miinfer-qwen3-decode-profile
scripts/run-m5b-profile.sh /path/to/Qwen3-8B-q4_0-b968826d.gguf
```

The retained `result.json` includes the model identity, build metadata,
per-category GPU/copy times, total dispatches, and unaccounted wall time.
`bench/results/<run-id>/` also retains the machine state and GPU telemetry.

## M5-C0 trace-free decode benchmark

`miinfer-qwen3-fast-decode-bench` measures the existing decode computation
without constructing or copying per-layer diagnostic traces. It retains the
same cache transitions and projection/precision policy, but copies only final
vocabulary logits to the host because greedy selection currently runs on the
CPU. The default workload uses token `14990`, warms eight generated tokens,
then measures 64 steady-state decode forward calls over five runs.

Run it through the same environment and telemetry runner:

```bash
cmake --preset mi50-release
cmake --build --preset mi50-release --target miinfer-qwen3-fast-decode-bench
scripts/run-m5c0-fast-decode.sh /path/to/Qwen3-8B-q4_0-b968826d.gguf
```

The JSON reports prefill, TTFT, decode, and total wall-time samples, the
complete generated ID sequence for the first run, model/build identity, and
whether trace copies were disabled. This is the A/B control for the first
optimization; it is not itself an optimization.
