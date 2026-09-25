# Prefill V2 Architecture Specification

## 1. Executive Summary

**Prefill V2** is a clean-sheet, performance-first prefill execution architecture designed exclusively for single-GPU inference on the **AMD Instinct MI50 32 GB (gfx906, Wave64)** running **Qwen3.8-27B-Q4_K_M**.

Prefill V2 completely decouples logical prompt length, physical matrix batch width, and internal recurrent causal chunk width. It replaces the legacy V1 multi-batch scheduler with an invariant semantic contract and explicit state flow.

---

## 2. Hardware & Model Contract

Prefill V2 is intentionally specialized and non-generic:

### 2.1 Hardware Contract
- **Device**: 1 × AMD Instinct MI50 (Vega 20 / gfx906)
- **Execution**: Native Wave64, 60 Compute Units, ~1 TB/s HBM2 memory bandwidth
- **Non-goals**: Multi-GPU, CUDA, RDNA, MI200/MI300, CPU offloading.

### 2.2 Model Contract: Qwen3.8-27B-Q4_K_M
- **Total Layers**: 64 (48 Gated DeltaNet recurrent layers, 16 full GQA attention layers in a repeating `3 × GDN + 1 × GQA` topology)
- **Hidden Dimension**: $D = 5120$
- **Intermediate (FFN) Dimension**: $D_{ffn} = 17408$
- **GDN Configuration**: 16 key/query heads ($H_k = 16$), 48 value heads ($H_v = 48$), head state dimension $S = 128$
- **GQA Configuration**: 24 Q heads ($H_q = 24$), 4 KV heads ($H_{kv} = 4$), head dimension $D_h = 256$
- **Quantization**: Native GGUF Q4_K_M tensor mix (Q4_K projections, Q5_K SSM output, Q6_K FFN down).

---

## 3. Core Architectural Invariants

### 3.1 Invariant Semantic Contract
In V1, varying batch widths ($B=512$, $B=128$, $B=64$, $B=4$) introduced divergent execution contracts, workspace assumptions, and numerical divergence.

In V2:
$$\text{Projection Batch Width} \neq \text{GDN Causal Chunk Width} \neq \text{Logical Prompt Length}$$

Logical model semantics are invariant. Changing physical GPU execution geometry never alters output semantics.

### 3.2 Fixed Internal GDN Chunk ($C = 64$)
GDN recurrence operates mathematically over fixed causal chunks of size $C = 64$:
- $N = 64 \implies 1 \times C64$
- $N = 128 \implies 2 \times C64$
- $N = 256 \implies 4 \times C64$
- $N = 512 \implies 8 \times C64$

The chunk size $C=64$ matches the gfx906 Wave64 execution model, maximizing LDS and register reuse while achieving proven mathematical equivalence to sequential recurrence.

### 3.3 Explicit State Flow
No hidden state transitions or global scheduler side-effects exist:
- **Matrix Recurrent State**: $S_t \in \mathbb{R}^{H_v \times S \times S} = [48, 128, 128]$ float32 (3.145 MB per layer).
- **Convolution History State**: $H_{conv} \in \mathbb{R}^{4 \times (2 \cdot D_{ssm})} = [4, 10240]$ float32 (163.84 KB per layer).
- State transitions are explicit:
```cpp
void forward(
    const float* d_input,
    float* d_output,
    const RecurrentLayerState& state_in,
    RecurrentLayerState& state_out,
    uint32_t start_pos,
    uint32_t token_count,
    hipStream_t stream
);
```

### 3.4 Zero Hot-Path Memory Allocation
- All scratch buffers are pre-allocated inside a monolithic `PrefillV2Workspace` (245.44 MiB for $N \le 512$).
- Zero device memory allocations (`hipMalloc`), reallocations, or host-device transfers occur during forward execution.

### 3.5 Resident Repacked Weights
- Weights are loaded from GGUF and repacked once into gfx906-native compact Mx MMQ tiles during layer initialization.
- Persistent footprint per recurrent layer: 246.88 MiB (Signature A: Q6_K QKV / Q6_K FFN-down) and 212.07 MiB (Signature B: Q4_K QKV / Q4_K FFN-down).
- Total for 48 recurrent layers: 10.76 GiB (saving ~3.54 GiB compared to older M23 tiling).
- No per-token or per-chunk conversions or repack operations in the hot path.

---

## 4. Recurrent Layer Dataflow

```text
Input Activation X [N, 5120]
  │
  ├──────────────────────────────────────────────────────┐ (Residual)
  ▼                                                      │
RMSNorm (attn_norm)                                      │
  │                                                      │
  ├────────────────────────┬────────────────────────┐    │
  ▼                        ▼                        ▼    │
QKV Proj (Q4_K MMQ)   Gate Proj (Q4_K MMQ)     Conv Prep │
[N, 10240]            [N, 6144]                ssm_conv1d│
  │                        │                        │    │
  │                        ▼                        │    │
  │                   Silu Gating                   │    │
  ▼                        │                        ▼    │
Chunkwise GDN (C=64) ◄─────┴─────────────────────────────┘
  │ (Input: Q, K, V, Beta, Alpha, S_in -> S_out)
  ▼
GDN Core Output [N, 6144]
  │
RMSNorm (ssm_norm)
  │
SSM Out Proj (Q5_K MMQ) [N, 5120]
  │
Residual Add ◄───────────────────────────────────────────┘
  │
  ├──────────────────────────────────────────────────────┐ (Residual)
  ▼                                                      │
RMSNorm (post_attn_norm)                                 │
  │                                                      │
  ├────────────────────────┬────────────────────────┐    │
  ▼                        ▼                        │    │
FFN Gate (Q4_K MMQ)   FFN Up (Q4_K MMQ)             │    │
[N, 17408]            [N, 17408]                    │    │
  │                        │                        │    │
  └───────────┬────────────┘                        │    │
              ▼                                     │    │
      SwiGLU Activation                             │    │
              ▼                                     │    │
      FFN Down (Q6_K MMQ) [N, 5120]                 │    │
              │                                     │    │
Residual Add ◄┴─────────────────────────────────────┘
  │
Final Output Activation Y [N, 5120]
```

---

## 5. Directory Structure & Namespace Separation

V2 is isolated cleanly from V1:

```text
include/miinfer/prefill_v2/
├── constants.hpp          # Hardware & Qwen3.8 architecture parameters
├── state.hpp              # RecurrentLayerState & RecurrentLayerStateStorage
├── workspace.hpp          # PrefillV2Workspace & scratch plan
├── kv_cache.hpp           # AttentionKvCacheView & AttentionLayerKvCacheStorage
├── recurrent_layer.hpp    # PrefillV2RecurrentLayer interface
├── attention_layer.hpp    # PrefillV2AttentionLayer interface
├── topology_block.hpp     # PrefillV2TopologyBlock (3xGDN + 1xGQA)
└── model.hpp              # PrefillV2Model (Full 64-layer pipeline)

src/prefill_v2/
├── state.cpp              # State allocation & lifecycle
├── workspace.cpp          # Workspace allocation & pointer partitioning
├── kv_cache.cpp           # Key/Value cache allocation & lifecycle
├── recurrent_layer.cpp    # Recurrent weight repacking & forward pipeline
├── attention_layer.cpp    # Attention weight repacking & forward pipeline
├── topology_block.cpp     # Ping-pong activation chaining across 4 layers
└── model.cpp              # Full 64-layer end-to-end model execution

bench/
├── prefill_v2_recurrent_layer_bakeoff.cpp  # Empirical bakeoff against V1 oracle
├── prefill_v2_topology_block_bakeoff.cpp   # 4-layer topology block bakeoff
└── prefill_v2_model_bench.cpp              # Full 64-layer multi-length benchmark
```

---

## 6. Qualification Ladder
 
1. **Slice 1 (Qualified)**: Single Recurrent Layer Vertical Slice & 3-Layer Chain.
   - V2-0001: Structurally qualified against canonical V1 token oracle ($5.31\times$ speedup at $N=512$).
   - V2-0002: Performance qualified with compact Mx MMQ against `FAST_V1` ($1.00\times$ speedup, matching $48.65\text{ ms}$ at $N=512$; $145.99\text{ ms}$ on 3-layer chain; $1.000000$ split-call invariant).
2. **Slice 2 (Qualified)**: Full 4-Layer Repeating Topology Block (`3 × GDN + 1 × GQA`).
   - V2-0003: Performance qualified at $180.89\text{ ms}$ at $N=512$ ($1.260\times$ faster than `FAST_V1`, $124.10\times$ faster than Oracle).
   - Exact bit-for-bit split-call invariant ($512$ vs $256 + 256$, max error $= 0.000000$, cosine $= 1.000000$).
3. **Slice 3 (Qualified)**: 64-Layer Full Model Prefill Pipeline (`PrefillV2Model`).
   - V2-0004: Integrated zero-spill Wave64 register-resident GDN scan (`launch_mx_gdn_chunk`).
   - Topology Block 0 latency dropped from $180.89\text{ ms} \to 145.89\text{ ms}$ ($1.564\times$ speedup over `FASTEST_V1`).
   - Full 64-layer end-to-end model prefill executed on single MI50.
   - P512 Execution: **2295.86 ms** (223.0 tok/s).
   - Stateful segmentation invariants pass across $512$ vs $256+256, 4\times 128, 8\times 64$ and continuation ($512+128$ vs $256+256+128$) with cosine $> 0.999$.
4. **Slice 4 (Qualified)**: Native P512 Macro Tiling & Refreshed Frontier Benchmark (V2-0005).
   - `kPrefillV2MacroTile = 512` physical macro tiling partitions arbitrary prompt lengths into canonical $N \le 512$ full-model passes with persistent states.
   - Sized monolithic workspace for $N=512$, reclaiming $\sim 900\text{ MiB}$ VRAM down to $297.44\text{ MiB}$ ($18.17\text{ GiB}$ total static VRAM including resident LM-Head, leaving $13.82\text{ GiB}$ free headroom).
   - Integrated resident Q6_K LM Head logit evaluation and greedy decoding.
   - Refreshed `mx-llama.cpp` baseline on identical hardware state (1606/1000 MHz, 225W): P64 is **1.346× WIN** ($613.19\text{ ms}$ vs $825.49\text{ ms}$), P128..P1024 achieves tight parity ($\le 1.6\%$ delta).
5. **Promotion**: Ready for default prefill engine promotion.
175: 
