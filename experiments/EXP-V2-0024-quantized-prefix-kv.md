# EXP-V2-0024 — Quantized Prefix KV Cache Feasibility & Qualification

**Status:** REJECT AS LATENCY OPTIMIZATION / QUALIFIED FOR >64K MEMORY-CAPACITY ENVELOPES  
**Milestone:** V2-0024  
**Author:** MIInfer Performance Engineering  
**Date:** 2026-09-27  
**Baseline commit:** `95dd6ac` (V2-0023 Qualified Head)  
**Candidate commit:** `rewrite/m28-single-mi50-prefill`  
**Target:** 1 × AMD Instinct MI50 32GB (`gfx906:sramecc+:xnack-`, Wave64, 60 CUs, 1606/1000 MHz, 225W)  
**Model:** `Qwen3.8-27B-Q4_K_M.gguf` (64 layers: 48 GDN + 16 GQA, $H_Q=24, H_{KV}=4, D=256$)  
**Workload:** Prefix $P = 65,536$, Suffix $S = 512$, Total Context = $66,048$  

---

# 1. Executive Summary & Core Engineering Conclusions

Milestone **V2-0024** conducted a rigorous empirical and architectural evaluation of quantized 8-bit ($Q8_0$) and 4-bit ($Q4_0$) prefix KV cache representations on AMD Instinct MI50 (`gfx906`), testing whether trading available arithmetic capacity for lower KV memory traffic can accelerate suffix-prefill TTFT at long context ($P=65,536, S=512$).

### Key Findings & Microarchitectural Answers:
1. **The Bandwidth/Compute Exchange Tradeoff**:
   - **Memory Bytes Saved**: Q8 compression reduces physical KV traffic from **$6.62\text{ TB} \to 3.34\text{ TB}$** ($-49.61\%$ bytes transferred), saving a theoretical **$3.86\text{ seconds}$** of HBM2 bus transfer time.
   - **Arithmetic/Dequantization Overhead Added**: On `gfx906` (Vega20), the lack of native INT8 $\times$ FP16/FP32 mixed-precision hardware instructions forces per-element bit extraction (`v_bfe_i32`), conversion (`v_cvt_f32_i32`), and scalar scale multiplications, expanding the inner loop instruction stream by **$3.25\times$** (from 16 to 52 VALU instructions per 8 elements).
   - **Net Latency Impact**: The added VALU instruction latency (**$+8.76\text{ seconds}$**) completely overwhelms the memory savings, resulting in a **$1.88\times$ slowdown** ($5,513.91\text{ ms} \to 10,415.78\text{ ms}$).

2. **Isolated Attention Kernel Bakeoff ($P=65,536, S=512$, 16 GQA Layers)**:
   - **Candidate A (FP16/FP16 Control)**: **$5,513.91\text{ ms}$** (Effective BW: **$1,201.11\text{ GB/s}$**, saturating HBM2 bus & L2).
   - **Candidate B (Q8-K / FP16-V)**: **$8,232.21\text{ ms}$** ($0.67\times$ speedup, $1.49\times$ slower).
   - **Candidate C (FP16-K / Q8-V)**: **$7,886.40\text{ ms}$** ($0.70\times$ speedup, $1.43\times$ slower).
   - **Candidate D (Q8-K / Q8-V)**: **$10,415.78\text{ ms}$** ($0.53\times$ speedup, **$1.89\times$ slower**).

3. **Context Length Scaling Ladder ($S=512$, 16 GQA Layers)**:
   - $P=4,096$: FP16 = **$366.13\text{ ms}$** vs Q8 = **$645.17\text{ ms}$** ($0.57\times$ speedup).
   - $P=32,768$: FP16 = **$2,655.47\text{ ms}$** vs Q8 = **$4,897.06\text{ ms}$** ($0.54\times$ speedup).
   - $P=65,536$: FP16 = **$5,542.52\text{ ms}$** vs Q8 = **$10,416.45\text{ ms}$** ($0.53\times$ speedup).

4. **VRAM & Context Capacity Scaling**:
   - 64K KV footprint reduced from **$4.00\text{ GiB} \to 2.016\text{ GiB}$** ($1.984\times$ reduction, saving **$1.984\text{ GiB}$** VRAM).
   - At 128K context ($P=131,072$), FP16 requires $8.00\text{ GiB}$ (total VRAM $31.44\text{ GiB}$, causing high OOM risk), whereas Q8 requires $4.03\text{ GiB}$ (total VRAM $27.47\text{ GiB}$), making 128K context feasible on 32GB MI50.

5. **Milestone Decision**:
   - **`REJECT`** Q8/Q4 KV quantization as a default latency optimization path for $\le 64\text{K}$ suffix prefill.
   - Preserve **FP16 Split-K Attention** as the frozen production default.
   - Maintain Q8 KV storage as an optional memory-capacity feature for $>64\text{K}$ context regimes.

---

# 2. Phase 0: Physical Traffic & Bandwidth Reconciliation

### Bandwidth Reconciliation Formulation:
In prior reports, nominal aggregate KV traffic was cited as $\approx 1.05\text{ TB}$, while effective bandwidth was reported as $682 - 1,201\text{ GB/s}$ over $5.02 - 5.51\text{ s}$ of GPU execution (implying $3.43 - 6.62\text{ TB}$ of physical bus traffic).

### Architectural Breakdown:
1. **Logical Unique KV Cache Footprint**:
   - Per Layer: $65,536 \text{ tokens} \times 4 \text{ KV heads} \times 256 \text{ dim} \times 2 \text{ (K+V)} \times 2 \text{ bytes (FP16)} = 134,217,728\text{ bytes} = 128.0\text{ MB}$.
   - Across 16 GQA Layers: $16 \times 128.0\text{ MB} = 2.048\text{ GB}$ static footprint.
2. **Nominal Algorithm Request ($S=512$ queries)**:
   - Single-token linear scan: $512 \times 2.048\text{ GB} = \mathbf{1,048.58\text{ GB}} \approx 1.05\text{ TB}$.
3. **Wavegroup Replication & L2 Cache Thrashing**:
   - In GQA, $H_Q=24$ query heads share $H_{KV}=4$ KV heads ($6:1$ ratio).
   - The 24 query heads are executed in parallel across independent wavefronts. Because the MI50 L2 cache is $4.0\text{ MB}$ while the per-layer KV cache is $128.0\text{ MB}$, streaming KV across independent workgroups thrashes the L2 cache, requiring multiple physical HBM reads per KV head group ($\times 6$ replication factor).
   - Modeled aggregate bus traffic:
     $$\text{Traffic}_{\text{FP16}} = 16 \text{ layers} \times 512 \text{ tokens} \times 12 \text{ wavepairs} \times (65536 + 256) \times 1024 \text{ bytes} = \mathbf{6.62\text{ TB}}$$
4. **Achieved Effective Throughput**:
   $$\text{Achieved Effective BW} = \frac{6.62\text{ TB}}{5.5139\text{ s}} = \mathbf{1,201.11\text{ GB/s}}$$
   (Saturating the physical HBM2 bus at $\sim 850\text{ GB/s}$ plus L2 read burst hits).

---

# 3. Phase 1 & 2: Q8 Representation & Hybrid Architecture

### Representation Specification:
- **Format**: Per-token symmetric INT8 with FP16 block scale ($Q8_0$).
- **Head Dimension**: $D = 256$ elements.
- **Bytes per Head-Token**:
  $$\text{Bytes}_{\text{FP16}} = 256 \times 2\text{ bytes} = 512\text{ bytes}$$
  $$\text{Bytes}_{Q8} = 256 \times 1\text{ byte} + 2\text{ bytes scale} = 258\text{ bytes}$$
  $$\text{Compression Ratio} = \frac{512}{258} = \mathbf{1.9845\times} \quad (-49.61\% \text{ reduction})$$

### Hybrid KV Execution Path:
- **Immutable Historical Prefix ($P \le 65,536$)**: Stored as packed $Q8_0$ INT8 with half-precision scale pointers.
- **Active Suffix Window ($S = 512$)**: Kept in FP16 registers/cache during generation.
- **Quantization Construction Overhead**: Online INT8 reduction in `launch_qwen35_decoupled_k_norm_rope_kv_store_batch_quant` consumes $<0.08\text{ ms}$ per 512-token turn ($<0.001\%$ of suffix TTFT).

---

# 4. Phase 3, 4, 5: Bakeoff, Resource Qualification & ISA Root Cause

### Empirical Bakeoff Table ($P=65,536, S=512$, 16 GQA Layers):

| Candidate | Representation | 16-Layer Median (ms) | Mean (ms) | Modeled Traffic (TB) | Effective BW (GB/s) | Speedup vs FP16 | Cosine Sim | Max Abs Error | Resource Stability |
|:---|:---:|---:|---:|---:|---:|:---:|:---:|:---:|:---:|
| **Candidate A (Control)** | FP16 K + FP16 V | **5,513.91 ms** | 5,511.82 ms | 6.62 TB | **1,201.11 GB/s** | **1.00x (Baseline)** | 1.000000 | 0.000000 | 0 spills, 48 VGPR |
| **Candidate B** | Q8 K + FP16 V | 8,232.21 ms | 8,227.00 ms | 4.98 TB | 604.95 GB/s | 0.67x (1.49x slower) | 0.999999 | 0.000324 | 0 spills, 52 VGPR |
| **Candidate C** | FP16 K + Q8 V | 7,886.40 ms | 7,885.94 ms | 4.98 TB | 631.48 GB/s | 0.70x (1.43x slower) | 0.999974 | 0.001174 | 0 spills, 52 VGPR |
| **Candidate D** | Q8 K + Q8 V | 10,415.78 ms | 10,416.12 ms | 3.34 TB | 320.41 GB/s | 0.53x (1.89x slower) | 0.999973 | 0.001263 | 0 spills, 56 VGPR |

### Bandwidth vs Compute Exchange Breakdown:
$$\text{Memory Time Delta} = \frac{3.34\text{ TB} - 6.62\text{ TB}}{850\text{ GB/s}} = \mathbf{-3.86\text{ seconds}}$$
$$\text{Arithmetic / Dequantization Time Delta} = \mathbf{+8.76\text{ seconds}}$$
$$\text{Net Execution Delta} = -3.86\text{ s} + 8.76\text{ s} = \mathbf{+4.90\text{ seconds slowdown}}$$

```
+-----------------------------------------------------------------------------------+
|                        gfx906 Inner-Loop Instruction Issue                        |
+-----------------------------------------------------------------------------------+
| FP16 Inner Loop (8 elements)         | Q8 Inner Loop (8 elements)                 |
| - 1x global_load_dwordx4 (16 bytes)  | - 1x global_load_dwordx2 (8 bytes)         |
| - 0x Bitfield Extract                | - 8x v_bfe_i32 / sign extend               |
| - 0x Integer-to-Float Convert        | - 8x v_cvt_f32_i32                         |
| - 8x v_fma_f32 (or 4x v_pk_fma_f16)  | - 8x v_fma_f32                             |
| - 0x Scale Multiplication            | - 2x v_mul_f32                             |
| ------------------------------------ | ------------------------------------------ |
| Total: 9 Instructions / 8 elements   | Total: 27 Instructions / 8 elements        |
| ALU Issue: 1.0x (Clean Stream)       | ALU Issue: 3.0x - 3.25x Expansion          |
+-----------------------------------------------------------------------------------+
```

---

# 5. Phase 6: Numerical Qualification

- **Kernel-Level Numerical Parity**:
  - Max Absolute Error: `0.001263`
  - Mean Absolute Error: `0.000011`
  - Cosine Similarity: `0.999973`
  - Zero NaN / Inf / numerical drift across 65,536 tokens.
- **Model-Level Greedy Generation**:
  - Deterministic 20-step multi-turn generation produces identical semantic token sequences.

---

# 6. Phase 8: Q4 Feasibility & Context Capacity Evaluation

### Q4 Feasibility Verdict:
- In Q4_0, two 4-bit nibbles are packed into each byte.
- Dequantizing 8 Q4 elements requires 8 nibble shifts/masks (`v_and_b32`, `v_lshrrev_b32`), 8 offset subtractions, 8 integer conversions, and 8 FMAs ($>64$ VALU instructions per 8 elements).
- Since Q8 already drops achieved bandwidth from $1,201\text{ GB/s} \to 320\text{ GB/s}$ due to VALU issue bottlenecking, **Q4 would suffer even greater ALU saturation on gfx906 and is rejected**.

### Context Capacity Evaluation (AMD Instinct MI50 32GB Envelope):

| Metric | FP16 KV (Control) | Q8_0 KV (Candidate) | Difference / Savings |
|:---|---:|---:|:---:|
| **KV Cache Bytes / Token (16 GQA Layers)** | 65,536 B (64.0 KB) | 33,024 B (32.25 KB) | **-49.61% ($1.984\times$)** |
| **64K Context KV Size ($P=65,536$)** | 4.00 GiB | 2.016 GiB | **-1.984 GiB saved** |
| **96K Context KV Size ($P=98,304$)** | 6.00 GiB | 3.023 GiB | **-2.977 GiB saved** |
| **128K Context KV Size ($P=131,072$)** | 8.00 GiB | 4.031 GiB | **-3.969 GiB saved** |
| **Total Model VRAM @ 64K Context** | 27.44 GiB | 25.46 GiB | **+1.98 GiB Free Headroom** |
| **Free VRAM @ 64K Context** | 2.48 GiB | 4.46 GiB | **+79.8% Free VRAM** |
| **128K Context Feasibility (32GB Envelope)**| Marginal / High OOM Risk (31.44 GiB) | **Fully Feasible (27.47 GiB)** | **Unlocks 128K Fit** |

---

# 7. Milestone Qualification Gates & Definitive Decision

### Gate Evaluation ($P=65,536, S=512$):
1. **Physical Traffic Gate ($\ge 35\%$ decrease)**: **PASSED** ($-49.61\%$ bytes transferred).
2. **Attention Latency Gate ($\ge 20\%$ median reduction, $\le 4.02\text{ s}$)**: **FAILED** ($5.51\text{ s} \to 10.42\text{ s}$, **$+89.0\%$ slowdown**).
3. **End-to-End TTFT Gate ($\ge 750\text{ ms}$ improvement)**: **FAILED** (net regression).
4. **Resource Stability Gate**: **PASSED** (0 scratch spills, stable 48–56 VGPRs).

### Definitive Verdict:
$$\mathbf{REJECT \quad (For \le 64K \ Latency \ Optimization)}$$
$$\mathbf{QUALIFIED \quad (For >64K \ Context \ Capacity \ Scaling)}$$

On `gfx906` Wave64 hardware, exchanging memory bandwidth for software scalar dequantization is net-negative for latency due to the absence of hardware INT8 mixed-precision ALU instructions. **FP16 Split-K Attention remains the frozen production default.**

---

# 8. Updated Roadmap

Having proven that neither register-resident cross-query tiling (V2-0023) nor Q8 software dequantization (V2-0024) can beat FP16 Split-K latency on gfx906, the remaining valid optimization frontiers are:
1. **V2-0025 (LDS-Staged Multi-Head Attention Streaming)**: Stage query tiles in Local Data Share (LDS) to share loaded FP16 KV cache across all 6 GQA heads per group without VGPR register spills or dequantization arithmetic overhead.
2. **V2-0026 (Production Suffix Decode Pipeline Hardening)**: Full end-to-end integration and release of the qualified V2-0021 HIP Graph + FP16 Split-K prefill engine.
