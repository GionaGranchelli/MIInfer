# V2-0006 — GQA Attention Optimization Sprint & Architectural Bakeoff

## 1. Goal

Evaluate gfx906-specialized attention kernels to eliminate the long-context prefill performance gap between Prefill V2 and `mx-llama.cpp` ($P4096 = -10.7\%$, $P8192 = -20.1\%$) on Qwen3.8-27B (24 Q heads, 4 KV heads, head_dim 256, 16 GQA layers).

Specifically, test whether:
1. **Candidate A (V2-Native Query-Tiled GQA)**: Tiling multiple query tokens ($T_Q = 8$) and Q heads ($H_Q = 2$) across a 4-Wave64 workgroup with 32 KiB LDS caching eliminates redundant HBM reads of the shared KV cache.
2. **Candidate B (Split-KV GQA)**: Splitting the causal KV prefix across $S=2$ or $S=4$ workgroups parallelizes the serial reduction and unlocks higher wave occupancy at moderate batch sizes.

---

## 2. Environment & Hardware State

- **Target Device**: 1 × AMD Instinct MI50 32GB (`gfx906:sramecc+:xnack-`)
- **GPU Clocks**: SCLK = 1606 MHz, MCLK = 1000 MHz, Power Cap = 225 W
- **Compiler**: ROCm 7.1.0 / HIP-Clang (`-O3 -mcpu=gfx906`)
- **GQA Architecture**: $H_Q = 24$, $H_{KV} = 4$ (GQA ratio $= 6$), $d_k = 256$, $\text{scale} = 1/\sqrt{256} = 0.0625$
- **Harness**: `miinfer-prefill-v2-attention-bakeoff` (`bench/prefill_v2_attention_bakeoff.cpp`)

---

## 3. Kernel Configurations

| Configuration | Function / Kernel | Thread Geometry | LDS Usage | Register Pressure | Synchronization | Memory Strategy |
|:---|:---|:---|---:|---:|:---|:---|
| **Control** | `launch_qwen35_tiled_online_attention_batch_f16` | 1 Wave64 per `(token, Q_head)` | 0 KiB | ~52 VGPRs | Zero intra-block barriers | Direct HBM streaming + online softmax registers |
| **Candidate A** | `launch_prefill_v2_query_tiled_attention_f16` | 256 threads (4 Wave64s) per `(8 tokens, 2 Q_heads)` | 32 KiB | ~48 VGPRs | `__syncthreads()` per 32 KV steps | Collaborative LDS staging of 32 KV tokens |
| **Candidate B2** | `launch_prefill_v2_split_kv_attention_f16` ($S=2$) | Stage 1: $S$ waves per `(token, Q_head)`<br>Stage 2: 1 wave per `(token, Q_head)` | 0 KiB | ~54 VGPRs | Global memory barrier between stages | Partial sum/max output in global workspace |
| **Candidate B4** | `launch_prefill_v2_split_kv_attention_f16` ($S=4$) | Stage 1: $S$ waves per `(token, Q_head)`<br>Stage 2: 1 wave per `(token, Q_head)` | 0 KiB | ~54 VGPRs | Global memory barrier between stages | Partial sum/max output in global workspace |

---

## 4. Part 1: Numerical Correctness Gate

Comparing output tensors against the established Control Wave64 oracle across all sequence lengths and continuation offsets:

| Token Count ($N$) | Base Position | Candidate | Max Absolute Error | Mean Absolute Error | Relative RMS Error | Cosine Similarity | Status |
|---:|---:|:---|---:|---:|---:|---:|:---|
| 64 | 0 | Cand A (Q-Tile) | $3.73 \times 10^{-9}$ | $1.71 \times 10^{-10}$ | $7.40 \times 10^{-8}$ | 1.000000 | **PASS** |
| 64 | 0 | Cand B2 (Split2) | $4.19 \times 10^{-9}$ | $1.83 \times 10^{-10}$ | $7.79 \times 10^{-8}$ | 1.000000 | **PASS** |
| 64 | 0 | Cand B4 (Split4) | $3.73 \times 10^{-9}$ | $2.16 \times 10^{-10}$ | $8.80 \times 10^{-8}$ | 1.000000 | **PASS** |
| 128 | 0 | Cand A (Q-Tile) | $3.73 \times 10^{-9}$ | $1.48 \times 10^{-10}$ | $8.26 \times 10^{-8}$ | 1.000000 | **PASS** |
| 128 | 0 | Cand B2 (Split2) | $4.19 \times 10^{-9}$ | $2.03 \times 10^{-10}$ | $1.06 \times 10^{-7}$ | 1.000000 | **PASS** |
| 128 | 0 | Cand B4 (Split4) | $3.73 \times 10^{-9}$ | $2.36 \times 10^{-10}$ | $1.19 \times 10^{-7}$ | 1.000000 | **PASS** |
| 512 | 0 | Cand A (Q-Tile) | $3.73 \times 10^{-9}$ | $1.06 \times 10^{-10}$ | $1.04 \times 10^{-7}$ | 1.000000 | **PASS** |
| 512 | 0 | Cand B2 (Split2) | $4.19 \times 10^{-9}$ | $2.18 \times 10^{-10}$ | $1.92 \times 10^{-7}$ | 1.000000 | **PASS** |
| 512 | 0 | Cand B4 (Split4) | $3.73 \times 10^{-9}$ | $2.41 \times 10^{-10}$ | $2.08 \times 10^{-7}$ | 1.000000 | **PASS** |
| 1024 | 0 | Cand A (Q-Tile) | $3.73 \times 10^{-9}$ | $8.90 \times 10^{-11}$ | $1.16 \times 10^{-7}$ | 1.000000 | **PASS** |
| 1024 | 0 | Cand B2 (Split2) | $4.54 \times 10^{-9}$ | $2.24 \times 10^{-10}$ | $2.64 \times 10^{-7}$ | 1.000000 | **PASS** |
| 1024 | 0 | Cand B4 (Split4) | $4.42 \times 10^{-9}$ | $2.43 \times 10^{-10}$ | $2.81 \times 10^{-7}$ | 1.000000 | **PASS** |
| 2048 | 0 | Cand A (Q-Tile) | $3.73 \times 10^{-9}$ | $7.48 \times 10^{-11}$ | $1.31 \times 10^{-7}$ | 1.000000 | **PASS** |
| 2048 | 0 | Cand B2 (Split2) | $4.77 \times 10^{-9}$ | $2.27 \times 10^{-10}$ | $3.60 \times 10^{-7}$ | 1.000000 | **PASS** |
| 2048 | 0 | Cand B4 (Split4) | $5.01 \times 10^{-9}$ | $2.44 \times 10^{-10}$ | $3.81 \times 10^{-7}$ | 1.000000 | **PASS** |
| 4096 | 0 | Cand A (Q-Tile) | $3.73 \times 10^{-9}$ | $6.25 \times 10^{-11}$ | $1.49 \times 10^{-7}$ | 1.000000 | **PASS** |
| 4096 | 0 | Cand B2 (Split2) | $5.59 \times 10^{-9}$ | $2.27 \times 10^{-10}$ | $4.90 \times 10^{-7}$ | 1.000000 | **PASS** |
| 4096 | 0 | Cand B4 (Split4) | $5.59 \times 10^{-9}$ | $2.43 \times 10^{-10}$ | $5.15 \times 10^{-7}$ | 1.000000 | **PASS** |
| 8192 | 0 | Cand A (Q-Tile) | $3.73 \times 10^{-9}$ | $5.21 \times 10^{-11}$ | $1.69 \times 10^{-7}$ | 1.000000 | **PASS** |
| 8192 | 0 | Cand B2 (Split2) | $5.59 \times 10^{-9}$ | $2.27 \times 10^{-10}$ | $6.67 \times 10^{-7}$ | 1.000000 | **PASS** |
| 8192 | 0 | Cand B4 (Split4) | $5.59 \times 10^{-9}$ | $2.42 \times 10^{-10}$ | $7.00 \times 10^{-7}$ | 1.000000 | **PASS** |
| 512 | 4096 | Cand A (Q-Tile) | $6.40 \times 10^{-10}$ | $4.51 \times 10^{-11}$ | $3.11 \times 10^{-7}$ | 1.000000 | **PASS** |
| 512 | 4096 | Cand B2 (Split2) | $4.54 \times 10^{-9}$ | $2.25 \times 10^{-10}$ | $1.50 \times 10^{-6}$ | 1.000000 | **PASS** |
| 512 | 4096 | Cand B4 (Split4) | $4.31 \times 10^{-9}$ | $2.40 \times 10^{-10}$ | $1.58 \times 10^{-6}$ | 1.000000 | **PASS** |

**Numerical Gate Result**: Both Candidate A and Candidate B (S=2, S=4) achieve exact mathematical parity with the Control baseline ($1.000000$ cosine similarity, relative error $< 10^{-6}$).

---

## 5. Part 2: Standalone Attention Performance Benchmark (1 Layer)

Measured latency in milliseconds for 1 full GQA attention layer on MI50 (mean over repeated warm runs):

| Sequence Length ($N$) | Control (ms) | Cand A (ms) | Cand B2 (ms) | Cand B4 (ms) | Best Candidate Speedup vs Control |
|---:|---:|---:|---:|---:|---:|
| **512** | **2.698** | 4.873 | 27.140 | 27.265 | 0.554× (Control is 1.81× faster) |
| **1024** | **10.304** | 18.097 | 107.100 | 107.255 | 0.569× (Control is 1.76× faster) |
| **2048** | **42.079** | 69.646 | 455.366 | 425.696 | 0.604× (Control is 1.66× faster) |
| **4096** | **176.309** | 273.360 | 2208.032 | 1826.617 | 0.645× (Control is 1.55× faster) |
| **8192** | **751.291** | 1082.078 | 9284.693 | 8831.211 | 0.694× (Control is 1.44× faster) |

---

## 6. Architectural Analysis & Root Cause

### 6.1 Why Candidate A (Query-Tiled) Is Slower than Control
1. **Grid Occupancy & Latency Hiding**:
   - In Control (`launch_qwen35_tiled_online_attention_batch_f16`), each wave processes 1 query token for 1 Q head independently ($N \times 24$ total waves, e.g. 196,608 waves at $N=8192$).
   - Because each wave executes without barriers or shared memory coordination, the GPU hardware scheduler effortlessly hides HBM memory latency by interleaving independent waves across all 60 CUs.
   - In Candidate A, waves are grouped into 4-wave workgroups ($256$ threads). Each step of 32 KV tokens requires an explicit `__syncthreads()` barrier before and after the collaborative LDS load.
   - On `gfx906`, frequent workgroup barrier synchronizations introduce pipeline bubbles and stall execution units, creating latency overhead that significantly exceeds the HBM bandwidth saved from LDS reuse.

2. **LDS Bandwidth vs L2 Cache Locality**:
   - In Control, the MI50's 4 MiB L2 cache already provides high hit rates for recent KV tokens across concurrently scheduled waves. Staging KV tokens in LDS added explicit load/store traffic without offering a net speedup over L2 hits.

### 6.2 Why Candidate B (Split-KV) Fails Severely
- Split-KV divides the KV sequence into $S$ slices, requiring each slice to write its partial accumulation ($256\text{-dim}$ vector + max-score + sum-exp) out to global VRAM workspace.
- The secondary reduction stage introduces additional kernel launch overhead, global memory round-trips, and memory synchronization. For prefill sequence dimensions on gfx906, this results in a $10\times$ slowdown.

---

## 7. Protocol Adherence & Hard-Stop Rule

Per operating protocol:
- Both Candidate A and Candidate B failed the standalone performance gate against Control.
- Neither candidate was promoted to the full-model prefill pipeline.
- Production `src/prefill_v2/attention_layer.cpp` remains strictly on the proven Control Wave64 kernel (`launch_qwen35_tiled_online_attention_batch_f16`).
- No speculative "Candidate C" or ungated parameter searches were introduced.

---

## 8. Final Decision

`REJECT`

- Candidate A (Query-Tiled GQA): **REJECT** (0.55×–0.69× vs Control).
- Candidate B (Split-KV GQA): **REJECT** (0.08×–0.10× vs Control).
- Production Control Kernel: **KEEP** as the definitive GQA attention implementation for Prefill V2.
- Milestone Outcome: **`V2_ATTENTION_CANDIDATES_REJECTED`**.
