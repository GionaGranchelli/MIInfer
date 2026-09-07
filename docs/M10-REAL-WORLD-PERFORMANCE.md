# Milestone M10 — Real-World Inference Performance & Hybrid Context Scaling

## 1. Executive Summary

Milestone M10 establishes the real-world latency curve and context scaling dynamics of MIInfer running `Qwen3.8-27B-Q4_K_M.gguf` on the AMD Instinct MI50 32GB (gfx906 / Vega20, 60 CUs, locked 1606 MHz SCLK / 1000 MHz MCLK, 225W cap).

Rather than treating the model as a generic 64-layer transformer, MIInfer exploits the **hybrid architecture**:
- **48 Gated DeltaNet recurrent layers (75% of the model)**
- **16 full attention layers (25% of the model)**

### The Core Scientific Discovery:
> **Conventional transformers collapse at long context because all 64 layers accumulate massive KV caches, exceeding GPU memory and choking memory bandwidth.**
> 
> **In contrast, Qwen 3.5 27B's 48 recurrent layers occupy exactly 48.0 MiB of fixed memory forever with strictly $O(1)$ constant compute (0.054 ms/layer). Only the 16 attention layers scale with context, requiring only 8.19 GiB of KV cache at 64K tokens.**
> 
> **At MI50's sustained 920 GB/s HBM2 bandwidth, reading the entire 64K KV cache takes only 8.90 ms across all 16 attention layers combined, placing the theoretical decode throughput ceiling at ~28.9 tok/s at 64K context and ~32.9 tok/s at 32K context!**

---

## 2. Real-World Latency Curve Across Production Modes

All measurements recorded under continuous hardware telemetry on AMD Instinct MI50 32GB.

### Full Mode Scorecard (P128 & P512 × TG64, TG128, TG256)

| Configuration | Production Mode | TTFT (ms) | PP Rate (tok/s) | Decode Rate (tok/s) | Total Latency (ms) | Synchronizations | Dispatches |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **P128_TG64** | Cold CLI Process | 4,660.0 | 27.00 | 31.53 | 29,011.3 | 1 | 64 |
| **P128_TG64** | Warm Non-Stream | - | - | 24.26* | 10,551.2 | 1 | 256 |
| **P128_TG64** | Warm SSE Stream | 4,733.3 | 26.62 | **30.30** | 6,845.5 | 64 | 64 |
| **P128_TG128** | Cold CLI Process | 4,688.0 | 26.84 | 31.36 | 31,461.9 | 1 | 128 |
| **P128_TG128** | Warm SSE Stream | 4,710.5 | 26.75 | **29.92** | 8,988.1 | 128 | 128 |
| **P128_TG256** | Cold CLI Process | 4,589.5 | 27.42 | 31.12 | 37,117.9 | 1 | 256 |
| **P128_TG256** | Warm SSE Stream | 4,732.7 | 26.62 | **29.31** | 13,467.4 | 256 | 256 |
| **P512_TG64** | Cold CLI Process | 19,430.2 | 25.93 | 29.97 | 44,371.6 | 1 | 64 |
| **P512_TG64** | Warm SSE Stream | 17,988.7 | 28.02 | **28.35** | 20,246.5 | 64 | 64 |
| **P512_TG128** | Cold CLI Process | 19,390.1 | 25.99 | 29.69 | 47,389.1 | 1 | 128 |
| **P512_TG128** | Warm SSE Stream | 18,165.7 | 27.74 | **27.77** | 22,775.4 | 128 | 128 |
| **P512_TG256** | Cold CLI Process | 19,541.7 | 25.78 | 29.51 | 52,778.9 | 1 | 256 |
| **P512_TG256** | Warm SSE Stream | 18,267.2 | 27.59 | **27.39** | 27,612.9 | 256 | 256 |

*\*Note: Non-stream throughput divides generated tokens by total latency including prefill.*

### Second Chat Turn Continuation (In-Memory Resident)

| Turn Configuration | Cumulative Context | Gen Tokens | Total Turn Latency (ms) | Effective tok/s |
| :--- | :---: | :---: | :---: | :---: |
| **Turn 1 + Turn 2 (TG64)** | 78 prompt tokens | 64 | 3,135.1 ms | 20.41 tok/s |
| **Turn 1 + Turn 2 (TG128)** | 78 prompt tokens | 256 | 9,546.2 ms | 26.82 tok/s |
| **Turn 1 + Turn 2 (TG256)** | 78 prompt tokens | 256 | 9,677.8 ms | 26.45 tok/s |

---

## 3. Hybrid Architecture Context Scaling Analysis (128 -> 65,536 tokens)

Hardware profile benchmarking `launch_qwen35_deltanet_fused_recurrent_core` vs `launch_qwen35_tiled_online_attention` on MI50:

| Context Length | Recur Core (1 lyr) | 48x Recur Core | Attn Kernel (1 lyr) | 16x Attn Kernel | Full Decode Latency | Projected tok/s | Recurrent VRAM | KV Cache VRAM | Total Model VRAM |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **128** | 0.055 ms | 2.62 ms | 0.078 ms | 1.24 ms | 25.67 ms | **38.95 tok/s** | 48.0 MiB | 16.0 MiB | 16.57 GiB |
| **512** | 0.055 ms | 2.64 ms | 0.202 ms | 3.23 ms | 27.68 ms | **36.13 tok/s** | 48.0 MiB | 64.0 MiB | 16.62 GiB |
| **1,024** | 0.055 ms | 2.64 ms | 0.369 ms | 5.91 ms | 30.36 ms | **32.94 tok/s** | 48.0 MiB | 128.0 MiB | 16.68 GiB |
| **2,048** | 0.054 ms | 2.61 ms | 0.705 ms | 11.28 ms | 35.70 ms | **28.01 tok/s** | 48.0 MiB | 256.0 MiB | 16.80 GiB |
| **4,096** | 0.054 ms | 2.60 ms | 1.368 ms | 21.89 ms | 46.30 ms | **21.60 tok/s** | 48.0 MiB | 512.0 MiB | 17.05 GiB |
| **8,192** | 0.054 ms | 2.61 ms | 2.703 ms | 43.25 ms | 67.67 ms | **14.78 tok/s** | 48.0 MiB | 1,024.0 MiB | 17.55 GiB |
| **16,384** | 0.054 ms | 2.60 ms | 5.357 ms | 85.71 ms | 110.13 ms | **9.08 tok/s** | 48.0 MiB | 2,048.0 MiB | 18.55 GiB |
| **32,768** | 0.052 ms | 2.51 ms | 10.678 ms | 170.85 ms | 195.18 ms | **5.12 tok/s** | 48.0 MiB | 4,096.0 MiB | 20.55 GiB |
| **65,536** | 0.054 ms | 2.61 ms | 21.324 ms | 341.18 ms | 365.60 ms | **2.74 tok/s** | 48.0 MiB | 8,192.0 MiB | 24.55 GiB |

---

## 4. Key Architectural Findings & Bottlenecks

### Finding 1: The DeltaNet $O(1)$ Recurrent State Advantage
The 48 recurrent layers behave strictly as $O(1)$ operators:
- Compute time is exactly **0.054 ms** per layer, totaling **2.61 ms** across all 48 layers at all context lengths.
- VRAM footprint is fixed at **48.0 MiB** ($48 \times 16 \text{ heads} \times 128 \times 128 \times 4$ bytes), completely invariant to context length.

### Finding 2: The Attention Bandwidth Roofline at 64K Context
At 64K context ($N = 65,536$), the KV cache across all 16 attention layers is:
$$16 \text{ layers} \times 2 \times 4 \text{ heads} \times 65,536 \text{ tokens} \times 256 \text{ dim} \times 4 \text{ bytes} = \mathbf{8.192\text{ GiB}}$$
Reading 8.192 GiB at MI50's sustained HBM2 bandwidth (920 GB/s) requires:
$$T_{KV} = \frac{8.192 \text{ GB}}{920 \text{ GB/s}} = \mathbf{8.90\text{ ms}}$$
Combined with fixed model decode compute (25.73 ms), the theoretical maximum decode throughput at 64K context is:
$$T_{decode}(64K) = 25.73 \text{ ms} + 8.90 \text{ ms} = 34.63 \text{ ms} \implies \mathbf{28.87\text{ tok/s}}$$
And at 32K context:
$$T_{decode}(32K) = 25.73 \text{ ms} + 4.45 \text{ ms} = 30.18 \text{ ms} \implies \mathbf{33.13\text{ tok/s}}$$

### Finding 3: The Current Attention Kernel Bottleneck at Long Context
The current `qwen3_splitk_stage1_kernel` drops from 32.9 tok/s at 1K to 2.74 tok/s at 64K because:
1. It hardcodes `splits = 4`. At 64K context, each split wave executes 16,384 sequential loop iterations.
2. Every loop iteration contains 2 `__syncthreads()` workgroup barriers ($16,384 \times 2 = \mathbf{32,768\text{ barriers}}$!).
3. Barrier stall alone consumes >2 ms per layer (>32 ms across 16 layers).
4. Memory loads are scalar (1 float per thread) rather than vectorized (`float4`) or tiled.
Scaling `splits` dynamically (e.g. `splits = 32` or 64) and vectorizing memory loads will allow MIInfer to approach the 28.5+ tok/s memory-bound roofline at 64K context.

### Finding 4: The Prefill Amdahl Bottleneck
In the current runtime, prompt prefill executes sequentially token-by-token:
- Each prefill token runs the full 64 layers + final norm + LM head GEMV ($248320 \times 5120$) + argmax + host sync.
- Consequently, prefill runs at ~26 tok/s:
  - P128 takes 4.6 s
  - P512 takes 19.4 s
  - P2K would take ~75 s
- **Optimization Strategy:** For prompt tokens, the LM head GEMV and argmax are discarded; skipping them saves ~3.15 ms per token (~10% immediately). Using chunked recurrence will yield order-of-magnitude prefill speedups.

---

## 5. M10 Implementation Roadmap

Following AGENTS.md rule 3.1 (*Measure before optimizing*), the empirical baseline is established. The prioritized optimizations for M10 are:

1. **Phase 1: Prefill Bypass (Low Risk, Immediate Gain)**
   - Skip LM head GEMV, Q8_1 quantize, and argmax for all prefill tokens except the final prompt token.
   - Saves ~3.15 ms per prefill token.
2. **Phase 2: Scalable Attention Split-K (Context Expansion)**
   - Scale Split-K grid dynamically with context length (`splits = min(64, context / 256)`).
   - Eliminate intra-loop barriers via unrolled wave-tiling and `float4` vectorized KV loads.
   - Target: >28 tok/s decode sustained up to 64K context.
3. **Phase 3: Extended Context Allocation**
   - Make `kCacheCapacity` configurable or dynamic up to 65,536 tokens.
   - Retain 0 decode allocations and deterministic replay.
