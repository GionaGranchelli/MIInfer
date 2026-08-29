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
output is JSON on standard output; `--json-output PATH` writes the same JSON to
a file while human-readable status is written to standard error.

The runner captures the environment before and after a benchmark:

```bash
scripts/run-bench.sh ./build/mi50-release/miinfer-bench \
  --warmup 5 --iterations 100
```

Results are stored under `bench/results/<run-id>/` as `environment-before.json`,
`result.json`, `telemetry.jsonl`, and `environment-after.json`. The runner
samples `rocm-smi` during the benchmark at 250 ms by default; set
`MIINFER_TELEMETRY_INTERVAL_MS` to change that interval or
`MIINFER_TELEMETRY=0` to disable it. Each telemetry line is a complete JSON
object, so clock and thermal drift can be inspected without a special parser.

The environment script records `UNAVAILABLE` when a command or metric is not
exposed by the local ROCm/Linux installation. The initial benchmark must not be
used to claim inference performance or to substitute for representative
model-shape experiments.
