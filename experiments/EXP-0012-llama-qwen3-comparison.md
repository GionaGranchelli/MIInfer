# EXP-0012 — Qwen3-8B Q4_0 MI50 comparison

## Hypothesis

The pinned gfx906-specialized llama.cpp implementation provides a useful
same-card performance control. Comparing it with MIInfer on the exact Qwen3-8B
Q4_0 artifact will distinguish a raw projection gap from a context-scaling or
execution-architecture gap.

## Scope

This is an external comparison, not an optimization candidate. No MIInfer
production code was changed. Standard `llama-bench` results and the closest
raw `hello` continuation workloads are reported separately because their timing
and correctness semantics differ.

## Environment

```text
GPU: AMD Instinct MI50 / gfx906, one visible device
Model: Qwen3-8B Q4_0
Model SHA256: 458634762bea7dbe19f3ce0614465bafd15ee90e815229c427043afcf195d628
Reference repository: milpster/gfx906-llama-cpp
Reference commit: 6e4ef6c1a553b8f61ad77bba18e9ca05aa677295
MIInfer commit: dd275325cfb9
ROCm: 7.1.52802-9999
HIP compiler: clang 20.0.0.rocm
```

The reference build used full GPU offload (`-ngl 99`), F16 K/V cache, 24
threads, and the pinned Q4_0 artifact. Active telemetry during the retained
reference run observed 1725 MHz SCLK and 1000 MHz MCLK. The MIInfer retained
64-forward run observed the same active clocks. A manual `rocm-smi
--setperflevel high` attempt was denied because this session has no sudo
password; no clock lock is claimed.

Reference raw outputs and telemetry are retained outside the repository at:

```text
/home/fedora-workstation/Development/mi50-artifacts/qwen3-8b-llama-comparison-20260831/
```

## Standard llama.cpp control

Command:

```bash
llama-bench \
  -m /path/to/Qwen3-8B-q4_0-b968826d.gguf \
  -p 512,0 -n 0,128,256 -r 5 -b 2048 -ub 512 -ngl 99 -t 24 -o jsonl
```

The combined run retained five raw samples for each case:

| Workload | Mean tok/s | Stddev | Raw tok/s |
|---|---:|---:|---|
| PP512 | 986.043 | 14.260 | 960.687, 992.501, 994.684, 992.033, 990.310 |
| TG128 | 91.080 | 0.735 | 89.8681, 90.9679, 91.4122, 91.3842, 91.7692 |
| TG256 | 91.558 | 0.478 | 91.8048, 91.8263, 91.8528, 91.5823, 90.7257 |

This establishes that the mature Q4_0 reference does not materially degrade
between TG128 and TG256 under its standard benchmark workload.

## Closest raw-token continuation controls

The shared input is the raw text `hello`, which the pinned tokenizer maps to
token `14990`, with full GPU offload and greedy sampling. The known eight-token
continuation is:

```text
8, 341, 286, 470, 330, 9707, 11, 330
```

### Short continuation

MIInfer trace-free control, result directory
`bench/results/20260831T201408Z-340101/`:

```text
warmup=1, measured decode forwards=7, iterations=3
decode mean: 225.763 ms / 7 = 31.006 tok/s
total mean: 255.785 ms, including prompt and eight generated tokens
generated IDs: 8,341,286,470,330,9707,11,330
```

Pinned `llama-simple`, `-n 8`, result retained in the external comparison
artifact:

```text
decoded: 8 tokens in approximately 0.16 s
reported speed: 50.35 tok/s
GPU eval: 154.79 ms / 8 = 51.68 tok/s
graphs reused: 7
generated text: ) {\n        return "Hello, "
```

The short comparison is approximately `1.63×` in llama.cpp's favor using the
reported decode speeds. The timing intervals are close but not identical:
MIInfer measures post-first-token forwards separately, while `llama-simple`
reports its own generated-token timer and also includes the prompt evaluation
in that outer interval.

### Growing-context continuation

MIInfer trace-free control, result directory
`bench/results/20260831T201752Z-342818/`:

```text
prompt tokens: 1
warmup: 0
measured decode forwards: 64
decode mean: 4435.321 ms / 64 = 14.430 tok/s
contexts exercised: approximately 1 through 65
```

Pinned `llama-simple`, `-n 64`, with 63 graph replays:

```text
decoded: 64 tokens in approximately 0.79 s
reported speed: 81.43 tok/s
GPU eval: 761.36 ms / 64 = 84.06 tok/s
graphs reused: 63
```

The close growing-context comparison is approximately `5.64×` in llama.cpp's
favor using the outer reported decode speeds. This is the important result:
MIInfer falls from roughly 31 tok/s to 14 tok/s as context grows, while the
pinned reference remains near 81 tok/s in the same raw-prompt regime.

The long runs are performance comparisons, not claims of identical generated
token IDs after the pinned eight-token prefix. Greedy trajectories can diverge
after a close numerical decision; the workload comparison remains useful
because both executions start from the same model and raw prompt and perform
incremental decode over a growing KV context.

## Interpretation

The standard reference benchmark confirms a mature Q4_0 gfx906 path around
91 tok/s for its TG128/TG256 control. The raw-token controls show that MIInfer
is closer on the short workload but has a major context-scaling deficit.

The existing M5-B profile measured 1,588 dispatches per token and 19.548 ms of
instrumented copy activity, but that profile is intrusive. The comparison does
not prove that all observed long-run gap is dispatch overhead; it does show
that context-dependent attention/KV work, repeated materialization, graph
capture, and dispatch architecture must be characterized before hand-tuning
FFN arithmetic.

## Decision

`KEEP` as the external performance control. No production optimization is
accepted by this experiment. M5-C1 should characterize the trace-free path's
dispatch/materialization behavior and context scaling, with one hypothesis per
subsequent A/B experiment.

## Follow-up

1. Add low-distortion dispatch and copy classification to the production-path
   benchmark, if possible without per-operation synchronization.
2. Measure decode at fixed contexts so attention/KV growth is separated from
   launch and allocation overhead.
3. Compare graph capture/replay or equivalent execution coarsening as the
   first architecture-level candidate.
4. Preserve the M5-C0 trace-free result as the MIInfer control for every A/B.
