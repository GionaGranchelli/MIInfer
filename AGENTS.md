# AGENTS.md

## Project

**MIInfer** is an experimental, performance-first LLM inference runtime targeting **AMD gfx906**, initially the **AMD Instinct MI50 32GB**.

The project exists to answer a specific engineering question:

> How much inference performance can be extracted from gfx906 when the runtime, memory layout, execution plan, and kernels are designed specifically for this architecture rather than treating it as a legacy backend of a generic inference framework?

MIInfer is **not** intended to become another general-purpose llama.cpp, vLLM, PyTorch, or generic ROCm runtime.

Specialization is a feature.

---

# 1. Mission

The initial target is intentionally narrow:

- AMD Instinct MI50 32GB
- gfx906 / Vega20
- Linux
- single GPU
- batch 1 / low batch
- one explicitly selected model family
- one explicitly selected quantization path
- local inference
- reproducible benchmarking
- correctness before optimization
- measurable performance improvements over the strongest available gfx906 llama.cpp baseline

Portability is not an initial goal.

Do not broaden project scope without explicit approval.

---

# 2. Core hypothesis

We are testing:

## H0

A purpose-built gfx906 runtime cannot materially outperform a well-optimized llama.cpp/gfx906 implementation.

## H1

Removing generic-runtime constraints allows meaningful performance improvements on MI50/gfx906.

The project must be developed in a way that makes this hypothesis falsifiable.

Performance claims require reproducible evidence.

---

# 3. Fundamental engineering rules

These rules take precedence over convenience.

## 3.1 Measure before optimizing

Never assume that an optimization is useful because:

- another gfx906 project uses it
- it is theoretically faster
- it reduces instructions
- it increases occupancy
- it worked on another model
- it worked on another AMD GPU
- it worked in CUDA
- it worked in another MI50 workload

Profile first.

Benchmark the exact workload.

Then optimize the measured bottleneck.

---

## 3.2 Every optimization needs a baseline

An optimization without an A/B comparison is not an optimization.

For performance-sensitive changes record:

- baseline commit
- candidate commit
- model
- model hash when practical
- quantization
- context length
- batch size
- ROCm version
- compiler version
- GPU
- GPU clocks
- HBM clocks where available
- power limit
- temperature
- VRAM use
- benchmark command
- repeated measurements
- mean / median where appropriate
- correctness result
- performance delta

Do not report single-run wins as conclusions.

---

## 3.3 Negative results are valuable

Do not delete failed experiments merely because they failed.

Record them under the experiment system.

A useful negative result prevents future contributors from repeating the same work.

Example:

```text
Hypothesis:
Explicit Y-tile prefetch will hide HBM latency.

Result:
-7.4% PP.

Reason:
The workload already kept Y sufficiently L2-resident; extra loads increased
instruction and register pressure.

Decision:
REJECT for this workload.
```

