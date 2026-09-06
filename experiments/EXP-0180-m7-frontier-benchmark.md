# EXP-0180 — M7 gfx906 Performance Frontier Benchmark

## Hypothesis

Specialized gfx906 llama.cpp forks implementing hardware-targeted weight repacking and fused epilogues (`mx-llama.cpp`) will outperform generic upstream llama.cpp on AMD Instinct MI50, establishing a higher performance frontier for `Qwen3.8-27B-Q4_K_M.gguf` than the pinned vanilla baseline beaten in EXP-0179.

## Motivation

In EXP-0179 (`aeaf1a2`), MIInfer achieved 23.33 tok/s (42.86 ms/tok) on TG64, successfully beating the project's historical reference baseline:
- Pinned vanilla `llama.cpp` (`c0bc8591`): 22.16 tok/s TG64 (45.13 ms/tok)
- Advantage: +5.3% throughput / -2.27 ms/tok.

Under Milestone M7, project discipline requires establishing the strongest reproducible baseline available on gfx906 rather than declaring victory over a dated or generic implementation. We must identify the true gfx906 performance frontier, measure all competitive candidates under qualified sustained hardware conditions, attribute performance deltas to specific architectural mechanisms, and set a new, falsifiable success gate for MIInfer.

## Competitive Candidates Evaluated

1. **Pinned Vanilla `llama.cpp` (`c0bc8591`, July 2026)**
   - Path: `/home/fedora-workstation/Development/mi50-artifacts/m6b0-llama-c0bc8591/bin/llama-bench`
   - Canonical MMQ / MMVQ execution, stock HIP graph capture.
2. **Current Upstream `llama.cpp` (`73a43d1f`, September 6, 2026)**
   - Path: `/home/fedora-workstation/Development/upstream-llama-build/bin/llama-bench`
   - Built with ROCm 7.1, gfx906 target (`-DGGML_HIP_ROCBLAS=ON`).
3. **`mx-llama.cpp` without repack (`2e9d29fe`, tag `b10904`)**
   - Path: `/home/fedora-workstation/Development/mx-llama-build/bin/llama-bench --no-repack 1`
   - Direct successor to `iacopPBK/llama.cpp-gfx906` with Qwen3.5 support.
   - Contains DPP warp reductions and GCN workgroup tunings.
4. **`mx-llama.cpp` with native repacking (`2e9d29fe`, tag `b10904`) [FRONTIER CANDIDATE]**
   - Path: `/home/fedora-workstation/Development/mx-llama-build/bin/llama-bench`
   - Custom `q8_repack` buffer frontend for gfx906 (K-quants repacked into de-aliased 2-plane layout).
   - Vectorized `dp4a` GEMV/GEMM + fused `{MM, MM, GLU}` dual-accumulator FFN epilogues.
5. **MIInfer EXP-0179 (`aeaf1a2`)**
   - Native Q4_K, Q5_K, and Q6_K Wave64 tiles + projection activation reuse.

## Environment & Hardware Qualification

- **GPU:** AMD Instinct MI50 32GB HBM2 (gfx906, Vega20, 60 CUs, Wave64)
- **Host CPU:** Intel(R) Xeon(R) CPU E5-2680 v3 @ 2.50GHz (24 threads)
- **Clock Configuration:** Locked MANUAL DPM
  - SCLK Level 7: 1606 MHz
  - MCLK Level 2: 1000 MHz
  - Power Cap: 225.0 W
- **Telemetry:** Continuous 250ms background sampling (`scripts/sample-gpu.sh`)
  - Total samples recorded: 2,238
  - SCLK 1606 MHz residency: 2,220 / 2,238 (99.2%)
  - MCLK 1000 MHz residency: 2,238 / 2,238 (100.0%)
  - Max junction temperature: 69.0 °C (well below 100 °C thermal throttle boundary)
- **Model:** `/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf` (17,095,778,304 bytes)

## Benchmark Methodology

- Automated benchmark harness: `scripts/run-m7-frontier-benchmark.sh`
- Tested across three token regimes: TG64, TG128, TG256
- 5 repetitions per candidate/regime with 5s inter-run thermal settle
- Median of runs reported

## Results

### Decode Throughput (tokens/second)

| Candidate | TG64 (tok/s) | TG128 (tok/s) | TG256 (tok/s) | Context Scaling (TG64 -> TG256) |
|---|---:|---:|---:|---|
| **Pinned Vanilla (`c0bc8591`)** | 22.16 | 22.54 | 22.56 | Flat (+1.8%) |
| **Upstream (`73a43d1f`)** | 22.46 | 22.53 | 22.60 | Flat (+0.6%) |
| **`mx-llama.cpp` (no repack)** | 22.95 | 23.06 | 23.05 | Flat (+0.4%) |
| **`mx-llama.cpp` (repack) [FRONTIER]** | **25.74** | **25.94** | **25.89** | **Flat (+0.6%)** |
| **MIInfer EXP-0179 (`aeaf1a2`)** | 23.33 | 22.72 | 21.70 | **Degrading (-7.0%)** |

### Per-Token Latency (ms/token)

| Candidate | TG64 (ms/tok) | TG128 (ms/tok) | TG256 (ms/tok) | Delta vs Frontier @ TG128 |
|---|---:|---:|---:|---:|
| **Pinned Vanilla (`c0bc8591`)** | 45.13 | 44.36 | 44.32 | +5.81 ms |
| **Upstream (`73a43d1f`)** | 44.53 | 44.38 | 44.25 | +5.83 ms |
| **`mx-llama.cpp` (no repack)** | 43.57 | 43.37 | 43.38 | +4.82 ms |
| **`mx-llama.cpp` (repack) [FRONTIER]** | **38.85** | **38.55** | **38.63** | **BASELINE (0.00 ms)** |
| **MIInfer EXP-0179 (`aeaf1a2`)** | 42.86 | 44.00 | 46.07 | **+5.45 ms** |

## Key Findings

1. **The gfx906 Performance Frontier is 25.94 tok/s (38.55 ms/tok), established by `mx-llama.cpp` (repack).**
2. Upstream `llama.cpp` (22.53 tok/s) has seen no material decode optimization for gfx906 since July 2026; competing only against upstream masks true hardware potential.
3. Repacking provides a massive +12.5% throughput boost (+2.88 tok/s / -4.82 ms/tok) over identical non-repacked kernels by de-aliasing weights and enabling vectorized `dp4a` arithmetic with fused `{MM, MM, GLU}` epilogues.
4. MIInfer EXP-0179 is faster than vanilla (+5.3%) and upstream (+3.9%) llama.cpp, but trails the frontier by 4.31 ms/tok at TG64 and 5.45 ms/tok at TG128.
5. MIInfer suffers from context degradation (-7.0% from TG64 to TG256), whereas `mx-llama.cpp` is perfectly flat across context lengths.

## Decision

**ESTABLISH NEW GOAL GATE.**
The performance gate for MIInfer is reset from the vanilla baseline (22.4 tok/s) to the qualified gfx906 frontier:
$$\text{Target TG64} \ge 25.94 \times 1.05 = \mathbf{27.24\ \text{tok/s}}\quad (\le \mathbf{36.71\ \text{ms/token}})$$
with context degradation eliminated at TG128 and TG256.
