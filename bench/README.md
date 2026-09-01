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

## M5-C1 position-scaled execution audit

`miinfer-qwen3-position-audit` characterizes the real trace-free decode path
at positions 1, 8, 16, 32, and 64. It reports clean production-path wall time
from a separate pass, plus deferred HIP-event timings for attention,
quantization, FFN projections, and all operation families. It also reports
dispatches, copy bytes, cache-write copy time, synchronization call sites, and
temporary allocations. Deferred timing avoids synchronizing every operation;
the audit pass still records events and must not be used as a throughput
benchmark.

Run it directly against the pinned model:

```bash
cmake --preset mi50-release
cmake --build --preset mi50-release --target miinfer-qwen3-position-audit
build/mi50-release/miinfer-qwen3-position-audit \
  /path/to/Qwen3-8B-q4_0-b968826d.gguf \
  --json-output /tmp/m5c1-position-audit.json
```

The retained M5-C1 result is documented in
`experiments/EXP-0013-qwen3-position-scaled-audit.md`. The first audit found
flat dispatch/copy/quantization/FFN costs and a strong position-dependent
cached-attention cost, making attention the next measured optimization target.

## M5-C2 cooperative cached attention

The production default now uses the M5-C2 cooperative cached-attention kernel:
one 256-thread workgroup per query head cooperates on score reductions,
softmax, and value accumulation. The existing serial implementation remains
available as the A/B control with `MIINFER_ATTENTION_KERNEL=serial`.

The candidate preserves the validated KV layout and greedy sequence. Its
measured trace-free controls improved from 31.006 to 40.267 tok/s on the short
workload and from 14.430 to 38.754 tok/s over 64 growing-context forwards.
The complete A/B record is in
`experiments/EXP-0014-cached-attention-parallel.md`.

## M5-C3 interleaved attention A/B benchmark

`miinfer-qwen3-attention-ab-bench` loads the model and execution plan once,
then alternates serial and cooperative cached-attention runs in balanced order
(`serial,parallel`, `parallel,serial`, ...). Each run resets the persistent KV
cache and measures the same trace-free prompt/warmup/decode workload. It
requires both policies to produce identical deterministic token IDs.

Run it with hardware-state capture:

```bash
cmake --preset mi50-release
cmake --build --preset mi50-release --target miinfer-qwen3-attention-ab-bench
scripts/run-m5c3-attention-ab.sh /path/to/Qwen3-8B-q4_0-b968826d.gguf
```

The clean M5-C3 run used three balanced pairs and 64 measured growing-context
decode forwards. Its raw result is retained under
`bench/results/20260901T061300Z-353809/`; the companion cooperative position
audit is under `bench/results/20260901T-m5c3-position-audit/`.

## M5-C4 post-attention baseline

M5-C4 reruns the trace-free cooperative path at the current clean commit and
extends the position audit through cache length 1024. The production benchmark
results are retained under `bench/results/20260901T062048Z-356125/` (short) and
`bench/results/20260901T062124Z-356897/` (64-token growing context). The
position audits are under `bench/results/20260901T0622-m5c4-position-audit/`
and `bench/results/20260901T0623-m5c4-long-audit/`.

The benchmark must be interpreted with the captured hardware state. The MI50
was in auto mode and telemetry observed approximately 925–930 MHz SCLK and
350 MHz MCLK, so these runs are not canonical replacements for the validated
1725/1000 MHz baseline. Use the interleaved M5-C3 harness for relative policy
comparisons and repeat M5-C4 after valid clock control is available.

## M5-C5a persistent decode workspace

The full decode path now allocates one `Qwen3GpuDecodeWorkspace` per decode
cache and reuses it across all layers and tokens. This removes per-token and
per-layer device-buffer allocation from the steady-state path while retaining
the existing kernels, launch topology, and precision policy. Static norm
weights are still uploaded during execution and are reserved for a separate
experiment.

The M5-C5a candidate results are retained under:

```text
bench/results/20260901T080644Z-365787/  short
bench/results/20260901T080718Z-366543/  64-token growing decode
bench/results/20260901T080500Z-364888/  position audit
```

At the observed 930/350 MHz auto-mode clocks, short decode improved from
40.528 to 53.625 tok/s and the 64-token growing workload improved from
38.094 to 48.334 tok/s. The position audit reports zero temporary allocations
at positions 1, 8, 16, 32, and 64; dispatches and copied bytes are unchanged.
These absolute rates remain hardware-state qualified until the validated
1725/1000 MHz clock state can be restored.

## M5-C5b resident normalization weights

The full decode path now reads immutable attention, Q/K, FFN, and final
normalization weights directly from the GPU plan's persistent weight arena.
This removes their redundant per-token host-to-device uploads while retaining
the C5a workspace, existing kernels, launch topology, and precision policy.

The clean C5b artifacts are retained under:

```text
bench/results/20260901T083234Z-371025/  short
bench/results/20260901T083307Z-371777/  64-token growing decode
bench/results/20260901T083350Z-372617/  position audit
```

At the observed 930/350 MHz auto-mode clocks, short decode reached 58.917
tok/s and the 64-token growing workload reached 52.791 tok/s. The audit
reports 2,082,304 copied bytes/token and 650 synchronization sites, down from
3,315,200 and 795 in C5a; dispatches remain 1,588 and temporary allocations
remain zero. These absolute rates remain hardware-state qualified until the
validated 1725/1000 MHz clock state can be restored. The full result is
documented in `experiments/EXP-0018-m5c5b-resident-norm-weights.md`.
