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
- All scratch buffers are pre-allocated inside a monolithic `PrefillV2Workspace` (247 MiB for $N \le 512$).
- Zero device memory allocations (`hipMalloc`), reallocations, or host-device transfers occur during forward execution.

### 3.5 Resident Repacked Weights
- Weights are loaded from GGUF and repacked once into gfx906-native MMQ tiles during layer initialization.
- Persistent footprint per recurrent layer: 307.03 MiB.
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
└── recurrent_layer.hpp    # PrefillV2RecurrentLayer interface

src/prefill_v2/
├── state.cpp              # State allocation & lifecycle
├── workspace.cpp          # Workspace allocation & pointer partitioning
└── recurrent_layer.cpp    # Weight repacking & forward pipeline

bench/
└── prefill_v2_recurrent_layer_bakeoff.cpp  # Empirical bakeoff against V1 oracle
```

---

## 6. Qualification Ladder

1. **Slice 1 (Current)**: Single Recurrent Layer Vertical Slice (Layer 0). Qualified against canonical V1 token oracle at $N \in \{64, 128, 512\}$.
2. **Slice 2**: Full 4-Layer Topology Block (`3 × GDN + 1 × GQA`).
3. **Slice 3**: 64-Layer Full Model Prefill Pipeline.
4. **Promotion**: V2 replaces V1 as default prefill engine only after end-to-end full-model qualification and baseline speedup verification.
