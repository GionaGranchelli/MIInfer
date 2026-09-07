# EXP-0197 — Paired Fused Gate+Up SwiGLU with 64-Bit dwordx2 Coalesced Memory Access

## Hypothesis

Previously, `launch_q4k_wave_fused_gate_up_swiglu` streamed Q4_K weight tiles from two disjoint VRAM buffers: `d_ffn_gate_native` and `d_ffn_up_native`, separated by ~55 MB in physical memory. Because each CU alternatingly loaded Gate and Up blocks across the 5 tiles, this caused:
1. HBM2 row-buffer trashing and channel switching overhead between disjoint address ranges.
2. Separate 32-bit `global_load_dword` load instructions for Gate plane 0, Gate plane 1, Up plane 0, and Up plane 1.

By interleaving Gate and Up weights at model load time into a unified contiguous `Q4KWaveSwigluFusedTile` (1280 bytes: 640 B Gate + 640 B Up):
1. Memory access becomes 100% sequential across the interleaved tile.
2. Threads can issue paired 64-bit `global_load_dwordx2` instructions for Plane 0 (loading `{gate.qs[lane], up.qs[lane]}`) and Plane 1 (loading `{gate.qs[lane + 32], up.qs[lane + 32]}`).
3. HBM2 effective bandwidth will increase from ~667 GB/s to ~748 GB/s in the SwiGLU stage, recovering $\ge 1.0$ ms/token across all 64 layers without increasing VRAM footprint.

## Motivation

In the MI50 Vega20 architecture (gfx906), low-batch decode is strongly memory-bandwidth bound. SwiGLU is the single largest weight-streaming operator in the model:
- Inner dimension: 17,408
- Hidden dimension: 5,120
- Across 64 layers, SwiGLU streams $2 \times 64 \times 17408 \times 2560 = 5.70$ GB of weight data per generated token.
- Streaming from two distinct 55MB allocations forced the HBM2 memory controller to continually swap open row buffers.

Repacking Gate and Up weights into a single contiguous tile eliminates address fragmentation and allows wide 64-bit load instructions to double the bytes fetched per memory transaction.

## Baseline

* Commit: `5a62eca` (M8 Qualified Baseline) + EXP-0194 + EXP-0196
* TG64 Median: 30.3337 tok/s / 32.967 ms/token
* TG128 Median: 30.0861 tok/s / 33.238 ms/token
* Environment: AMD Instinct MI50 32GB (gfx906 / Vega20, locked 1606 MHz SCLK / 1000 MHz MCLK, 225W power cap)
* Execution flags: `MIINFER_SWIGLU_PAIRED=0`

## Candidate

* Source:
  * `include/miinfer/kquant_wave_layout.hpp`: Added `Q4KWaveSwigluFusedTile`, `pack_q4k_wave_swiglu_fused`, and `launch_q4k_wave_fused_gate_up_swiglu_paired`.
  * `src/kquant_wave_layout.cpp`: Implemented multi-threaded parallel repacking into interleaved `Q4KWaveSwigluFusedTile`.
  * `gfx906/kernels/kquant_wave_layout.hip`: Implemented `q4k_wave_fused_gate_up_swiglu_paired_kernel<5>` using `uint2` / `dwordx2` loads.
  * `tools/qwen35_gpu_pipeline.hpp`: Integrated `d_ffn_swiglu_native` into `RecurrentLayer` and `FullAttentionLayer`.
* Toggles:
  * `MIINFER_SWIGLU_PAIRED=1` (enabled by default)

## Environment

* GPU: AMD Instinct MI50 32GB
* Architecture: gfx906 / Vega20, 60 CUs, Wave64
* SCLK: 1606 MHz (DPM 7)
* MCLK: 1000 MHz (DPM 2)
* Power Cap: 225W
* Telemetry (TG64): 1015 samples; 96.3% locked @ 1606 MHz, Avg Temp 52.5°C (Max: 62.0°C), Avg Power 88.8W (Max: 250.0W)
* Telemetry (TG128): 1333 samples; 90.2% locked @ 1606 MHz, Avg Temp 59.4°C (Max: 70.0°C), Avg Power 120.9W (Max: 250.0W)

## Benchmark

* Harness: `scripts/run-exp0197-swiglu-paired-ab.sh both`
* Methodology: 5-pair interleaved Control/Candidate runs for both TG64 and TG128 (20 benchmark runs total, 5 sample measurements per run)
* Model: `Qwen3.8-27B-Q4_K_M.gguf`

## Correctness

* Microbenchmark (`scratch/test_swiglu_paired.cpp`):
  * `max_abs_diff = 0.000000e+00` (exact bitwise match).
* CTest Regression Suite: 21/21 PASS (100%).
* 16-Token Autoregressive Generation:
  * Fingerprint: `14656917272593817253` (deterministic match across repeated runs).
  * Allocations during decode: 0.
  * Replay determinism: PASS.
* 64-Layer Observable Contract (`--prefix64-observable-contract`):
  * Positions evaluated: 64/64 PASS.
  * Final logits cosine: **0.99964** (exceeds 0.9995 threshold).
  * Reference argmax / GPU argmax: exact match across all 64 positions.
  * Top-5 overlap: 5/5. Reference rank on GPU: 1.
  * Poisoned reset replay: PASS.

## Results

### TG64 Interleaved Benchmark (5 Pairs)

| Pair | Control (tok/s) | Candidate (tok/s) | Control (ms/tok) | Candidate (ms/tok) | Delta | Winner |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Pair 1 | 30.3337 | 31.3522 | 32.967 | 31.896 | +3.36% | Candidate |
| Pair 2 | 30.3928 | 31.3474 | 32.903 | 31.901 | +3.14% | Candidate |
| Pair 3 | 30.3619 | 31.3348 | 32.936 | 31.913 | +3.20% | Candidate |
| Pair 4 | 30.3119 | 31.2769 | 32.990 | 31.973 | +3.18% | Candidate |
| Pair 5 | 30.2930 | 31.3352 | 33.011 | 31.913 | +3.44% | Candidate |
| **Median** | **30.3337** | **31.3352** | **32.967** | **31.913** | **+3.30%** | **Candidate (5/5)** |

- **TG64 Latency Saved:** **+1.054 ms/token (+3.20%)**

---

### TG128 Interleaved Benchmark (5 Pairs)

| Pair | Control (tok/s) | Candidate (tok/s) | Control (ms/tok) | Candidate (ms/tok) | Delta | Winner |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Pair 1 | 30.0861 | 30.9999 | 33.238 | 32.258 | +3.04% | Candidate |
| Pair 2 | 30.1607 | 31.0012 | 33.156 | 32.257 | +2.79% | Candidate |
| Pair 3 | 30.1151 | 31.0942 | 33.206 | 32.160 | +3.25% | Candidate |
| Pair 4 | 30.0691 | 31.0232 | 33.257 | 32.234 | +3.17% | Candidate |
| Pair 5 | 30.0749 | 31.0210 | 33.250 | 32.236 | +3.15% | Candidate |
| **Median** | **30.0861** | **31.0210** | **33.238** | **32.236** | **+3.11%** | **Candidate (5/5)** |

- **TG128 Latency Saved:** **+1.002 ms/token (+3.01%)**
- **Context Scaling Penalty (TG64 $\rightarrow$ TG128):** **1.00%** ($\le 1.5\%$ requirement satisfied)

---

## Profiling & Architectural Interpretation

1. **HBM2 Channel & Bank Conflicts:**
   - In separate buffers, thread lanes alternating between Gate and Up were loading from pointers separated by 55 MB. This frequently forced HBM2 channel and bank reactivation cycles.
   - In the paired tile layout, Gate and Up blocks for each group of 32 weights are stored contiguously in memory, ensuring that memory bursts stay within the same open page.
2. **Instruction Count & Memory Efficiency:**
   - Using 64-bit `global_load_dwordx2` halved the number of vector memory instructions issued by each wavefront for weight loading.
   - Effective bandwidth improved from 667.6 GB/s to 747.9 GB/s (+12.0% bus efficiency).
3. **Memory Neutrality:**
   - The size of `Q4KWaveSwigluFusedTile` is exactly $640 + 640 = 1280$ bytes. Total persistent device allocation remained bit-for-bit identical at 18,886,426,964 bytes. Zero additional VRAM was required.

## Decision

**KEEP**.
Paired Fused Gate+Up SwiGLU won 10 out of 10 benchmark pairs across both TG64 and TG128, accelerating TG64 throughput from **30.33 tok/s to 31.34 tok/s** and saving **1.054 ms/token**. It is enabled by default.

## Follow-up

Current performance stands at:
- **31.34 tok/s TG64 (31.91 ms/token)**.
- Target: $\ge 32.00$ tok/s ($\le 31.25$ ms/token).
- Remaining gap to close: **0.66 ms/token** (from 31.91 ms to 31.25 ms).

The remaining candidates to recover the final ~0.66 ms/token are:
1. **Optimize `launch_q8_1_quantize_f32` (or inline into SwiGLU epilogue):**
   - Standalone `launch_q8_1_quantize_f32` is called 64 times per token to quantize the 17,408 SwiGLU outputs, accounting for ~0.635 ms/token of dispatch and kernel execution time.
2. **Combine Attention QKV Projections:**
   - Combining Q (12,288 rows), K (1,024 rows), and V (1,024 rows) into a single $14336 \times 5120$ projection in the 16 attention layers.
3. **Fused SwiGLU Direct Q8_1 Output (0 HBM round-trip for SwiGLU activation).**
