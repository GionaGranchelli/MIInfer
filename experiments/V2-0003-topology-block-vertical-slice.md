# V2-0003 — Prefill V2 Slice 2: 4-Layer Repeating Topology Block (3 × GDN + 1 × GQA Attention)

## Hypothesis

Integrating the complete 4-layer repeating topology block (`Block 0 = Layers 0, 1, 2 GDN Recurrent + Layer 3 GQA Attention`) into Prefill V2 on single-MI50 (gfx906, Wave64) with resident compact Mx MMQ weights, unified monolithic workspace, dedicated FP16 KV cache storage, and decoupled RoPE/KV contracts will:
1. Deliver exact numerical stability and cosine similarity $\ge 0.999$ against the canonical sequential V1 Token Oracle across all evaluated prompt lengths ($N \in \{64, 128, 512\}$).
2. Outperform the qualified production-shaped $B512$ `FAST_V1` baseline by $\ge 1.20\times$ on Block 0 at $N=512$ ($\le 190\text{ ms}$ vs $227.88\text{ ms}$).
3. Achieve perfect bit-for-bit stateful split-call invariant equivalence ($512$ vs $256 + 256$) with cosine similarity $= 1.000000$ and max absolute error $= 0.000000$ across all intermediate recurrent states, KV caches, and final block outputs.
4. Fit the complete 64-layer model (16 topology blocks) into single-MI50 VRAM at $\sim 17.51\text{ GiB} / 32\text{ GiB}$ total static footprint.

## Motivation

Slice 1 (V2-0001 / V2-0002) successfully qualified the single recurrent layer with $C=64$ internal GDN chunking and compact Mx MMQ projections.

However, the real Qwen3.8-27B model consists of a 4-layer repeating topology block:
$$\text{Block}_k = 3 \times \text{GDN Recurrent Layers} + 1 \times \text{Full GQA Attention Layer}$$
To advance Prefill V2 toward full-model qualification, Slice 2 must construct the multi-layer pipeline:
- Ping-pong activation chaining ($d\_ping \leftrightarrow d\_pong$) across sequential layers without intermediate copies or hot-path allocations.
- Full GQA Attention layer with decoupled Q, K, and V projections, per-head RMSNorm, rotary position embeddings (RoPE), online tiled causal softmax attention with sigmoid gating, and FFN.
- Monolithic shared workspace (~297.4 MiB) accommodating both recurrent activations (GDN chunking scratch, QKV/gate/conv/beta/decay) and attention activations (Q+gate, RoPE query, K/V projections, tiled attention shared tiles).
- Dedicated per-layer FP16 Key/Value Cache storage ($H_{kv}=4, D_h=256$, capacity=32768 tokens, 128 MiB/layer).

## Environment

- **GPU**: 1 × AMD Instinct MI50 32 GB (gfx906, Vega 20, 60 CUs, Wave64)
- **ROCm / Compiler**: ROCm 6.2+ / hipcc (LLVM 18)
- **Model**: `Qwen3.8-27B-Q4_K_M.gguf` (`/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf`)
- **Evaluated Topology**: Block 0:
  - Layer 0 (GDN Recurrent): Q6_K QKV, Q4_K Gate, Q5_K SSM-out, Q4_K FFN-g/u, Q6_K FFN-down (246.88 MiB)
  - Layer 1 (GDN Recurrent): Q6_K QKV, Q4_K Gate, Q5_K SSM-out, Q4_K FFN-g/u, Q6_K FFN-down (246.88 MiB)
  - Layer 2 (GDN Recurrent): Q6_K QKV, Q4_K Gate, Q5_K SSM-out, Q4_K FFN-g/u, Q6_K FFN-down (246.88 MiB)
  - Layer 3 (GQA Attention): Q4_K Q+Gate, Q4_K K, Q4_K V, Q4_K Out, Q4_K FFN-g/u, Q4_K FFN-down (223.84 MiB)

## Baseline vs Candidate

- **Oracle**: Canonical sequential V1 token loop (`RecurrentLayer::run` + `FullAttentionLayer::run`) across all 4 layers — correctness oracle.
- **FAST_V1**: Qualified production-shaped B512 Mx prefill baseline (`RecurrentLayer::prefill_wide` + `FullAttentionLayer::prefill_wide` with `MIINFER_PREFILL_WIDE_MX_REPACKED_MMQ=1`).
- **Candidate (Prefill V2)**: Clean-sheet `PrefillV2TopologyBlock` chaining $3 \times \text{PrefillV2RecurrentLayer} + 1 \times \text{PrefillV2AttentionLayer}$ with resident compact Mx MMQ weights, monolithic shared workspace, and explicit recurrent states + KV cache.

## Memory Footprint & Static VRAM Budget

| Component | Per 4-Layer Block | Projected 64-Layer Model (16 Blocks) |
| :--- | :--- | :--- |
| **Persistent Layer Weights** | 964.47 MiB | **15.070 GiB** |
| **Recurrent States ($48 \text{ layers}$)** | 9.45 MiB ($3 \text{ layers}$) | **151.500 MiB** |
| **Attention KV Caches ($16 \text{ layers}$, 32K capacity)** | 128.00 MiB ($1 \text{ layer}$) | **2.000 GiB** |
| **Shared Monolithic Workspace** | 297.44 MiB | **297.438 MiB** (shared by all layers) |
| **Total Static VRAM Footprint** | **1.399 GiB** | **17.508 GiB / 32.000 GiB** |

Total persistent device memory for the entire 64-layer Qwen3.8-27B model fits with 14.49 GiB of remaining headroom on a single MI50 32 GB GPU.

## Empirical Results

### 1. Mandatory $N=512$ Topology Block Baseline Comparison

| Unit / Topology | V1 Token Oracle (ms) | FAST_V1 (ms) | Prefill V2 (ms) | Speedup vs FAST_V1 | Speedup vs Oracle |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **Block 0 ($3 \times \text{GDN} + 1 \times \text{GQA}$)** | 22,449.06 ms | 227.96 ms | **180.89 ms** | **1.260× (+20.7% faster)** | **124.10×** |

### 2. Multi-Length Scaling ($N=64, 128, 512$)

| Prompt Length ($N$) | V1 Token Oracle (ms) | FAST_V1 (ms) | Prefill V2 (ms) | Speedup vs Oracle | Throughput | Block Output Cosine |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| **N = 64** | 3,127.18 ms | N/A (unsupported) | **42.88 ms** | **72.92×** | 1,492 tok/s | **0.999018** |
| **N = 128** | 5,694.25 ms | N/A (unsupported) | **53.77 ms** | **105.91×** | 2,381 tok/s | **0.999196** |
| **N = 512** | 22,449.06 ms | 227.96 ms | **180.89 ms** | **124.10×** | 2,830 tok/s | **0.999544** |

### 3. V2 Block 0 Layer Breakdown ($N=512$, Total = 180.88 ms)

| Layer | Type | Execution Time (ms) | Fraction of Block Time |
| :--- | :--- | ---: | ---: |
| **Layer 0** | GDN Recurrent | 48.74 ms | 26.95% |
| **Layer 1** | GDN Recurrent | 48.73 ms | 26.94% |
| **Layer 2** | GDN Recurrent | 48.82 ms | 26.99% |
| **Layer 3** | GQA Attention | 34.59 ms | 19.12% |

### 4. Stateful Split-Call Invariant Test ($512$ vs $256 + 256$)

| Tensor / State | Cosine Similarity | Max Absolute Error | Relative RMS | Invariant Status |
| :--- | ---: | ---: | ---: | :--- |
| **Block Output ($512 \times 5120$)** | **1.000000** | **0.000000** | **0.000000** | **EXACT MATCH** |
| **Layer 0 Recurrent State** | **1.000000** | **0.000000** | **0.000000** | **EXACT MATCH** |
| **Layer 1 Recurrent State** | **1.000000** | **0.000000** | **0.000000** | **EXACT MATCH** |
| **Layer 2 Recurrent State** | **1.000000** | **0.000000** | **0.000000** | **EXACT MATCH** |
| **Layer 3 FP16 Key Cache** | **1.000000** | **0.000000** | **0.000000** | **EXACT MATCH** |
| **Layer 3 FP16 Value Cache** | **1.000000** | **0.000000** | **0.000000** | **EXACT MATCH** |

### 5. Full 64-Layer Model Latency Projection

| Execution Path | Projected P512 Latency | Comparison vs Target |
| :--- | ---: | :--- |
| **FAST_V1 Baseline (64 Layers)** | 3,647.36 ms (3.65 s) | 1.579× of mx-llama.cpp |
| **Prefill V2 (16 Blocks × 180.89 ms)** | **2,894.25 ms (2.89 s)** | **1.253× of mx-llama.cpp** |
| **mx-llama.cpp Target Baseline** | ~2,310.00 ms (~2.31 s) | 1.000× (Reference Target) |

## Decision

**KEEP — QUALIFIED (`V2_TOPOLOGY_BLOCK_QUALIFIED`)**.

Prefill V2 Slice 2 establishes:
1. Seamless multi-layer ping-pong chaining ($3 \times \text{GDN} + 1 \times \text{GQA}$).
2. Exact stateful mathematical invariance across split boundaries ($512$ vs $256+256$, max error $= 0.0$).
3. $1.260\times$ speedup over qualified FAST_V1 on Block 0 (20.7% faster).
4. Full static VRAM footprint of 17.51 GiB / 32 GiB, projecting full-model P512 prefill latency of 2.89 s.
