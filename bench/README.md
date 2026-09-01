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

## M5-C6a execution-overhead attribution

The C5b position audit provides the current copy and synchronization
breakdown without changing the production path. Per decode token it reports
576 KV-cache writes (`294,912` bytes), 72 layer input/output D2D copies
(`1,179,648` bytes), and one final logits D2H copy (`607,744` bytes). The
copy-call subtotal is 649 synchronization sites; the deferred profile adds
one finalization synchronization, for 650 measured synchronization events.
Dispatch topology remains 1,588/token. The audit and decision are recorded in
`experiments/EXP-0019-m5c6a-execution-overhead-audit.md`.

The C6b direct handoff result is recorded below. The next isolated candidate
after C6b is coalesced KV-cache writing.

## M5-C6b direct layer-output handoff

The trace-free fast path now writes the final FFN residual directly into the
next ping-pong buffer. Trace-producing execution retains the scratch buffer and
copy behavior. The explicit control is selected with
`MIINFER_LAYER_OUTPUT_HANDOFF=copy`; `direct` is the production default.

Run the balanced A/B benchmark with:

```bash
cmake --preset mi50-release
cmake --build --preset mi50-release --target miinfer-qwen3-attention-ab-bench
scripts/run-m5c6b-layer-output-ab.sh /path/to/Qwen3-8B-q4_0-b968826d.gguf
```

The benchmark uses the existing interleaved harness in `--mode layer-output`.
The clean C6b result is retained under
`bench/results/20260901T085927Z-374528/` and is documented in
`experiments/EXP-0020-m5c6b-direct-layer-output-handoff.md`. It reports
`copy,direct,direct,copy,copy,direct` ordering, identical deterministic IDs,
and 50.896937 versus 50.896418 tok/s at the observed 930/350 MHz clocks.
The position audit reports the exact structural reduction from 72 to 36 layer
I/O copies, from 2,082,304 to 1,492,480 copied bytes/token, and from 650 to
614 synchronization sites including finalization. The neutral timing result
is retained as such; the cleanup is kept for its simpler ownership contract.

## M5-C6c coalesced KV-cache writes

The production path now appends one token's K/V vectors with one device-side
store launch per layer, preserving the validated FP32 cache layout and append
ordering. The per-head memcpy control is selected with
`MIINFER_KV_CACHE_WRITE=copy`; the coalesced `store` path is the default.

Run the balanced A/B benchmark with:

```bash
cmake --preset mi50-release
cmake --build --preset mi50-release --target miinfer-qwen3-attention-ab-bench
scripts/run-m5c6c-kv-cache-ab.sh /path/to/Qwen3-8B-q4_0-b968826d.gguf
```

The clean result is retained under
`bench/results/20260901T091213Z-377343/` and documented in
`experiments/EXP-0021-m5c6c-coalesced-kv-cache-writes.md`. With balanced
`copy,store,store,copy,copy,store` ordering, throughput improved from
51.758283 to 54.600468 tok/s, while the pinned 64-forward IDs remained
identical. The candidate position audit reports KV memcpy calls and bytes
falling from 576/294,912 to 0/0, synchronization sites falling from 614 to 38,
and dispatches rising from 1,588 to 1,624. Rates are qualified by the observed
930/350 MHz auto-mode clocks.

## M5-C6d GPU-side greedy argmax

The trace-free greedy API keeps vocabulary logits on the GPU, runs a
deterministic first-index argmax reduction, and copies only the selected
`uint32_t` token ID. The existing full-logit API remains available for
diagnostics. The A/B harness compares both paths in balanced order:

```bash
cmake --preset mi50-release
cmake --build --preset mi50-release --target miinfer-qwen3-attention-ab-bench
scripts/run-m5c6d-argmax-ab.sh /path/to/Qwen3-8B-q4_0-b968826d.gguf
```

The clean result is retained under
`bench/results/20260901T093604Z-383152/` and documented in
`experiments/EXP-0022-m5c6d-gpu-argmax.md`. The metadata-clean balanced
64-forward run measured 54.7099 tok/s for the full-logit control and 54.9812
tok/s for GPU-argmax at the observed 930/350 MHz clocks, with identical
generated IDs.
The GPU-argmax position audit is under
`bench/results/20260901T093647Z-384099/`; it reports final result traffic
falling from 607,744 to 4 bytes, total copied bytes falling from 1,197,568 to
589,828, and dispatches rising from 1,624 to 1,625. Temporary allocations
remain zero and synchronization sites remain 38.

## M5-C7 post-copy-cleanup profile

The C6d position audit now also records a lightweight whole-token HIP event
around an otherwise trace-free decode. This is separate from the detailed
per-operation deferred profile, whose event-recording overhead makes its
summed GPU times unsuitable for direct comparison with clean wall time.

The P1/P64 profile is retained under
`bench/results/20260901T094450Z-386101/` and documented in
`experiments/EXP-0023-m5c7-post-copy-cleanup-profile.md`. Clean wall versus
whole-token GPU time was 15.313/15.475 ms at P1 and 19.780/19.928 ms at P64.
The path remains at 1,625 dispatches, 589,828 copied bytes, 38
synchronization sites, and zero temporary allocations per token. Detailed
timing identifies FFN projection as the largest individual family at about
7.0 ms at P64; quantization, normalization, and conversion account for 1,118
of the dispatches. The resulting next target is a measured FFN kernel
experiment, not HIP graph capture.

## M5-C9a production FFN attribution

The position audit now includes measurement-only FFN stage counters while
retaining the production trace-free execution path. It reports separate GPU
event totals and dispatch counts for FFN normalization, Gate/Up/Down input
quantization, Gate/Up/Down projection, SwiGLU, and the final FFN residual.
The aggregate profile continues to report full-token categories, copies,
synchronizations, and allocations. Deferred event timing is attribution data;
use the separate production wall measurement for throughput.

Run the P1/P64 attribution used by EXP-0027 with:

```bash
cmake --preset mi50-release
cmake --build --preset mi50-release --target miinfer-qwen3-position-audit
build/mi50-release/miinfer-qwen3-position-audit \
  /path/to/Qwen3-8B-q4_0-b968826d.gguf \
  --positions 1,64 \
  --gpu-argmax \
  --json-output /tmp/m5c9a-position-audit.json
```

The measurement-only result is recorded in
`experiments/EXP-0027-m5c9a-ffn-attribution.md`. At the observed 930/350 MHz
auto-mode clocks, FFN totals were 9.880 ms at P1 and 9.979 ms at P64 in the
deferred profile. The SwiGLU plus Down-input quantization chain was 0.795 ms
at P64 across 108 dispatches, making it a bounded candidate for a separate
fused experiment. Gate and Up also independently quantize the same normalized
input; that reuse opportunity remains separate from C9b.

The existing interleaved A/B harness can compare the separate and fused
SwiGLU paths on the same warmed growing-context workload:

```bash
build/mi50-release/miinfer-qwen3-attention-ab-bench \
  /path/to/Qwen3-8B-q4_0-b968826d.gguf \
  --mode swiglu-fusion --warmup 8 --generated-tokens 64 --pairs 3
```

The fused path is selected only by `MIINFER_SWIGLU_Q8_FUSION=fused` inside
that A/B mode; the default production path remains `separate`.

EXP-0028 rejected the candidate for production selection: its standalone
Q8Exact payload test passed and the local chain was 33.8% faster, but the
full fixed-prefix decode diverged at position 38. The separate path remains
the production default. See
`experiments/EXP-0028-m5c9b-swiglu-q8-fusion.md`.

## M5-C9c Gate/Up activation-Q8 reuse

The C9c candidate checks whether Gate and Up independently quantize the same
FFN-normalized input, then reuses one unchanged Q8 buffer for both GEMVs. The
candidate is selected only in the fast path with
`MIINFER_FFN_Q8_REUSE=shared`; the production default is `shared`, and
`separate` remains an explicit regression/A/B control.

The interleaved A/B harness is:

```bash
scripts/run-m5c9c-gate-up-q8-ab.sh \
  /path/to/Qwen3-8B-q4_0-b968826d.gguf
```

For real-model byte-level verification, run the position audit with the
candidate and verifier enabled:

```bash
MIINFER_FFN_Q8_REUSE=shared \
MIINFER_VERIFY_FFN_Q8_REUSE=1 \
build/mi50-release/miinfer-qwen3-position-audit \
  /path/to/Qwen3-8B-q4_0-b968826d.gguf \
  --positions 1,8,16,32,64 --gpu-argmax \
  --json-output /tmp/m5c9c-position-audit.json
```

The verifier recreates the old separate Gate/Up quantization into a second
persistent buffer and compares payload plus metadata byte-for-byte. It passed
180 checks across positions 1, 8, 16, 32, and 64 with zero mismatches. It is
diagnostic-only and adds host-visible copies; it must not be used for timing.
The balanced three-pair A/B produced identical 64-token trajectories and
measured 54.5501 tok/s separate versus 55.1724 tok/s shared (+1.14%). C9c is
therefore production-selected; see
`experiments/EXP-0029-m5c9c-gate-up-q8-reuse.md`.

## M5-C10a refreshed P64 production profile

The C10a measurement-only profile reruns the position audit after C9c with
shared Gate/Up Q8 reuse enabled. The P64 result is retained under
`bench/results/20260901T134000Z-m5c10a/` and documented in
`experiments/EXP-0030-m5c10a-refreshed-profile.md`.

At the observed approximately 930/350 MHz clocks, P64 measured 20.213 ms
clean wall, 20.121 ms whole-token GPU time, and 27.828 ms deferred attribution.
The path reports 1553 dispatches, 38 synchronization sites, zero temporary
allocations, and 589,828 bytes of residual copy accounting. The largest
families are FFN projection (6.963 ms), attention (4.935 ms), normalization
(2.978 ms), LM head (2.876 ms), and conversion (2.278 ms). C10a changes no
production behavior; it only refreshes the ranking for the next experiment.

## M5-C10b normalization/conversion boundary attribution

The C10b measurement-only audit adds named deferred HIP-event stages for
normalization, F32/F16 conversion, projection-input Q8 quantization, and the
final norm-to-Q8_K boundary. It runs the real trace-free path at positions
1/8/16/32/64 with shared Gate/Up Q8 reuse and does not change production
behavior. The clean-commit P64 result is retained under
bench/results/20260901T-c10b-boundary-clean/ and documented in
experiments/EXP-0031-m5c10b-normalization-conversion-attribution.md.

At P64 the clean wall/whole-token GPU event was 19.602/19.733 ms, deferred
attribution was 27.277 ms, and the path reported 1553 dispatches, 38
synchronization sites, zero allocations, and 589,828 bytes of copy
accounting. The named boundary map shows repeated small normalization and
materialization stages rather than a new copy or allocation pathology. The
attention F32→F16→F32 boundary remains semantically required; C10b therefore
selects one bounded FFN RMSNorm + norm-scale + F32→F16 + shared-Q8 experiment
for the next slice, with byte-identical Q8 and full trajectory gates.

## M5-C10c FFN normalization-to-shared-Q8 fusion

C10c implemented an opt-in one-dispatch FFN RMSNorm + norm-scale + exact
F32→F16 + shared Q8 candidate. The clean real-model verifier recorded 180/180
FP16 and 180/180 Q8 checks with zero mismatches, and Release CTest remained
19/19. The three serial five-iteration A/B pairs are retained under
`bench/results/20260901T-c10c-ab-clean/`: control averaged 54.125905 tok/s and
the fused candidate 51.302002 tok/s, a 5.217% regression. The candidate reduced
P64 total dispatches 1553→1445 and FFN normalization/Q8 dispatches 144→36,
but its one-workgroup processing took about 2.182 ms versus 1.423 ms for the
separate stages. It is therefore rejected for production selection; the
separate path remains the default and the candidate remains diagnostic-only.

## M5-C8a FFN projection shape characterization

The kernel-only `miinfer-q4-q8-gemv-bench` characterization is recorded in
`experiments/EXP-0024-m5c8a-ffn-shape-characterization.md`, with raw artifacts
under `bench/results/20260901T102000Z-389000/`. At the observed approximately
930/350 MHz clocks, the current production-like Q4_0 × Q8_1 controls measure
about 50.24 µs for Gate, 50.24 µs for Up, and 57.28 µs for Down; all repeated
runs pass the quantized oracle. The existing Wave64 control is only marginally
faster for Gate/Up and the available alternate Down geometry is slower, so no
production selection changed. C8b remains a new, measured FFN geometry
candidate rather than a speculative promotion.

## M5-C8b Down four-Wave64 candidate

The candidate experiment is recorded in
`experiments/EXP-0025-m5c8b-down-four-wave64.md`, with raw artifacts under
`bench/results/20260901T104000Z-391000/`. Four independent Wave64 reductions
per 256-thread workgroup passed the Q4_0 × Q8_1 oracle but regressed the
long-K Down shape from 57.28 µs to 83.36–84.80 µs. Gate and Up also regressed
about 1.9%. The candidate is retained as an opt-in diagnostic and is not
production-selected; this geometry family is rejected for C8c.

## M5-C11a production and llama.cpp differential baseline

C11a refreshes the accepted production path after C10c: shared Gate/Up Q8
reuse is enabled, while the rejected normalization/Q8 and SwiGLU/Q8 fusions
are disabled. The fast decode benchmark uses prompt ID `14990`, eight warmup
tokens, 64 measured forwards, five iterations, and three strictly serial
runs. The retained artifacts are under
`bench/results/20260901T-c11a-minfer/`; the P1/P64 production audit is under
`bench/results/20260901T-c11a-position-audit/20260901T193605Z-415258/`.

The three MIInfer runs are deterministic and average 55.0778 tok/s (1161.996
ms for 64 forwards), with the same 64-token sequence and Release CTest 19/19.
At P64 the clean wall/whole-token GPU event is 19.797/19.924 ms; deferred
attribution is 27.778 ms. The largest category is FFN projection at 6.956 ms,
followed by attention at 4.937 ms, quantization at 3.340 ms, normalization at
2.947 ms, LM head at 2.876 ms, and conversion at 2.270 ms. The path remains
at 1,553 dispatches, 38 synchronization sites, zero temporary allocations,
and 589,828 bytes of residual copy accounting.

The fresh pinned gfx906 llama.cpp `llama-bench` control is retained outside the
repository under
`/home/fedora-workstation/Development/mi50-artifacts/qwen3-8b-llama-c11a-20260901/`.
It measures 984.552 PP512 tok/s, 91.875 TG128 tok/s, and 91.692 TG256 tok/s
over five samples. Its telemetry reached approximately 1725/1000 MHz during
measurement, while MIInfer's measured phase remained approximately 925/350
MHz, so the fresh 1.67x TG differential is directional rather than a fair
clock-matched claim. The prior exact raw-token 64-forward control remains in
EXP-0012 at 81.43 tok/s.

C11a is measurement-only. The next bounded experiment is an exact-shape
FFN GEMV differential against the strongest available gfx906 MMVQ path; no
new fusion, geometry sweep, or graph capture is selected without direct
evidence.

## M5-C11b exact-shape FFN GEMV differential

C11b closed the exact-shape FFN differential without selecting a new kernel.
The retained direct comparison in `experiments/EXP-0008-direct-mmvq.md` found
MIInfer approximately tied or faster than the pinned gfx906 MMVQ path for
Gate, Up, and Down: -2.08%, -2.08%, and -11.29%, respectively. MMVQ's clear
historical advantage was K/V, which was already addressed by the accepted
EXP-0009 128-thread K/V geometry.

The current-tree sequential sanity runs measured 52.48 µs for Gate and 52.80
µs for Up, with the quantized oracle passing. A concurrent sanity run was
discarded as contaminated. The current benchmark's synthetic Down mode does
not select the production Down path, so it was not used as a production Down
result.

The clock qualification was subsequently resolved with the stock
`profile_peak`/`stable_peak` mode. MIInfer measured 55.356 tok/s with active
telemetry at 1725/1000 MHz; the pinned llama.cpp control measured 90.566
TG128 and 90.389 TG256 tok/s under the same peak state. C11b therefore stops
blind FFN/MMVQ porting and publishes a controlled end-to-end differential. See
`experiments/EXP-0034-m5c11b-exact-shape-ffn-differential.md`.

## M5-C12a stable-peak non-FFN profile

C12a refreshed the accepted shared-reuse production path at stable peak after
C11b. The current Release build measured 55.419 tok/s over 64 decode forwards.
The P64 position audit measured 19.579 ms clean wall time and 19.605 ms for the
whole-token GPU event, with 27.179 ms of deferred category attribution. The
path remained at 1553 dispatches, 38 synchronization sites, zero temporary
allocations, and 589,828 residual copy bytes.

At P64, the largest non-FFN categories were attention (4.928 ms), quantization
(3.184 ms), normalization (2.835 ms), conversion (2.147 ms), and Q/K/V
preparation (1.642 ms). Attention was the only category that grew with
position: it increased from 0.451 ms at P1 to 4.928 ms at P64, closely matching
the 4.455 ms clean-wall increase. The pinned llama.cpp shape control measured
90.817 tok/s for PP1/TG64 at the same stable-peak clocks. This is a shape
control, not a token-identical comparison. See
`experiments/EXP-0035-m5c12a-stable-peak-non-ffn-profile.md`.

C12a selects one bounded C12b target: differential profiling of cooperative
cached attention at P64 and longer contexts. No FFN GEMV, clock, generic
fusion, or dispatch-count experiment is selected from this profile alone.

## M5-C12b cooperative attention scaling

C12b extended the accepted cooperative attention path to P64, P128, P256,
P512, and P1024 at stable-peak clocks. Attention measured 4.932, 9.491,
18.607, 36.775, and 74.071 ms respectively. The production wall increased
19.665 to 88.678 ms over the same range, while dispatches stayed at 1553,
syncs at 38, allocations at 0, and residual copy accounting at 589,828 bytes.
The scaling is linear and shows no second context-collapse defect.

One opt-in four-Wave64 history-partitioned candidate was implemented under
`MIINFER_ATTENTION_KERNEL=history`. It changed the first generated token from
`8` to `8673`, so it failed the autoregressive correctness gate and was
rejected without production selection. The candidate changed reduction order;
the existing parallel attention path remains the default. See
`experiments/EXP-0036-m5c12b-cooperative-attention-scaling.md`.

## M5-C13a fixed-cost floor profile

C13a re-profiled the accepted shared Gate/Up Q8 reuse production path at
stable-peak clocks and short positions `1,2,4,8`. The raw audit is retained at
`bench/results/20260901T-c13a-floor/position-audit.json` and the decision is
documented in `experiments/EXP-0037-m5c13a-fixed-floor-profile.md`.

| Position | Wall ms | Whole-token GPU ms | Attention ms | Wall minus attention ms |
|---:|---:|---:|---:|---:|
| 1 | 15.032 | 15.071 | 0.450 | 14.582 |
| 2 | 15.342 | 15.186 | 0.522 | 14.820 |
| 4 | 15.431 | 15.425 | 0.663 | 14.768 |
| 8 | 15.712 | 15.656 | 0.949 | 14.763 |

The fixed wall-minus-attention component is therefore approximately 14.7
ms/token, consistent with the C12b P64/P1024 endpoints. Whole-token GPU event
time tracks clean wall time closely, while dispatches remain 1553/token,
synchronizations 38/token, allocations zero, and residual copy accounting
589,828 bytes/token. At P1 the largest named categories are FFN projection
(6.969 ms), quantization (3.179 ms), LM head (2.943 ms), normalization
(2.856 ms), and conversion (2.168 ms). Category timings are deferred event
attribution and overlap; they are not additive wall-clock components.

C13a changes no production behavior. FFN projection was not selected because
C11b already cleared the exact Gate/Up/Down shapes against the pinned gfx906
MMVQ path. C13b then audited the proposed exact-shape LM-head Q6_K×Q8_K
differential before any production implementation change.

## M5-C13b LM-head contract audit

C13b audited the proposed exact-shape LM-head differential and found that the
pinned llama.cpp gfx906 MMVQ implementation maps Q6_K to `vec_dot_q6_K_q8_1`,
not Q8_K. MIInfer's production LM head uses Q6_K×Q8_K. The two paths therefore
do not share an exact activation representation or dot-product contract, so a
direct kernel-latency comparison would be invalid.

The live MIInfer stable-peak P1 recheck measured 2.936 ms for its LM-head
stage; C13a's retained value is 2.943 ms with one dispatch. The external
whole-token PP1/TG64 context control measured 90.446 tok/s over three samples,
but it is not used to attribute LM-head time. No Q8_1 compatibility path,
production kernel change, or C13c target was selected. See
`experiments/EXP-0038-m5c13b-lm-head-contract-audit.md`.
