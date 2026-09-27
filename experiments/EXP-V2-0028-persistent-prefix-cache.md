# EXP-V2-0028 — Persistent Prefix Cache & Session Restore Qualification

## 1. Executive Summary

Milestone **V2-0028** qualifies durable cross-process state persistence and restoration for MIInfer on the AMD Instinct MI50 (gfx906, 60 CUs, 32GB HBM2).

Prior to V2-0028, prefix state reuse was strictly resident in GPU VRAM / process memory (`ReusableContext`). While in-memory reuse delivered ~2.3 s TTFT for suffix ingestion ($S=512$), process restarts, server reboots, agent session resumption, or server scale-to-zero forced complete re-execution of expensive cold prefill (~19.5 s for 4K tokens, ~210 tok/s).

V2-0028 implements durable, bit-exact disk serialization and restoration for the dual-state hybrid architecture:
1. **48 GDN Recurrent SSM States**: Fixed 151.5 MiB (48 layers $\times$ 3.0 MiB state matrix + 48 layers $\times$ 160 KiB 1D-conv histories).
2. **16 GQA Attention KV Caches**: Linear in prefix length $P$ ($P \times 16 \text{ layers} \times 8 \text{ heads} \times 128 \text{ dim} \times 2 \text{ bytes (FP16)} \times 2 (\text{K}+\text{V}) = 64\text{ KB/token}$). At $P=4096$, this is exactly 256.0 MiB.
3. Total serialized footprint at $P=4096$: **407.52 MiB**.

### Qualification Outcome
- **Correctness / Parity Gate**: **PASSED**. 100% token-for-token deterministic parity between cold baseline prefill and persistent session restore across 32 generated tokens.
- **I/O Efficiency**: Disk write = 574.4 ms (~0.7 GB/s); Disk restore = 491.2 ms (~0.8 GB/s).
- **Time-to-First-Token (TTFT)**: Dropped from **22,112.05 ms** (cold) to **3,111.87 ms** (disk restore + suffix prefill), delivering a **7.1x speedup** end-to-end including disk I/O.
- **Decision**: **KEEP & QUALIFY**. Integrated into `PrefillV2Model` and `miinfer serve --session-dir PATH`.

---

## 2. Hypothesis & Architectural Motivation

### Hypothesis
Durable disk serialization of both recurrent SSM states (48 layers) and attention KV caches (16 layers) can be restored into GPU memory in sub-second latency, bypassing expensive multi-second cold ingestion while preserving 100% numerical and token generation parity.

### Hardware & Workload Context
- **GPU**: AMD Instinct MI50 32GB HBM2 (gfx906, Vega20, 60 CUs, 1725 MHz engine clock, 1000 MHz HBM2 clock).
- **Model**: `Qwen3.8-27B-Q4_K_M.gguf` (64 layers: 48 GDN SSM + 16 GQA Attention, hidden dimension 5120).
- **Workload**: Prefix length $P = 4096$ tokens, Suffix length $S = 512$ tokens, Decode length = 32 tokens.

---

## 3. Implementation Details

### 3.1 Serialization Format (`.miinfer`)
Defined in [`include/miinfer/prefill_v2/persistent_session.hpp`](../include/miinfer/prefill_v2/persistent_session.hpp):

```cpp
#pragma pack(push, 1)
struct PersistentSessionHeader {
    char magic[4] = {'M', 'I', 'S', 'S'}; // MIInfer Session State
    std::uint32_t version = 20260928;
    char model_id[64] = {};
    char quantization[32] = {};
    std::uint32_t prefix_length = 0;
    std::uint64_t token_hash = 0;
    std::uint32_t gdn_layers = 48;
    std::uint64_t gdn_state_bytes = 0;
    std::uint32_t gqa_layers = 16;
    std::uint8_t kv_quant_mode = 0;
    std::uint64_t kv_bytes = 0;
    std::uint64_t tokens_bytes = 0;
    std::uint64_t created_timestamp = 0;
};
#pragma pack(pop)
```

Payload layout:
`[Header (168 B)] [Tokens (P * 4 B)] [GDN Layer States (48 * 3.156 MiB)] [GQA KV Caches (16 * P * 4096 B)]`

### 3.2 KV Cache Buffer Raw Access
Extended `AttentionLayerKvCacheStorage` in [`src/prefill_v2/kv_cache.cpp`](../src/prefill_v2/kv_cache.cpp):
- `download_raw(void* dst_host, std::size_t token_count, hipStream_t stream)`: Copies contiguous K and V active regions from device VRAM to host buffer.
- `upload_raw(const void* src_host, std::size_t token_count, hipStream_t stream)`: Uploads host KV state into device VRAM and sets cache active length to `token_count`.

### 3.3 Engine & Server Integration
- Added `save_session()`, `load_session()`, and `restore_matching_session()` to `PrefillV2Model`.
- Integrated `persistent_session_dir` into `GenerateOptions`:
  - When non-empty, checks disk cache for matching prefix sequences before cold prefill.
  - Automatically persists qualifying prefixes ($\ge 256$ tokens) to disk immediately after prefill completes.
- Integrated `--session-dir PATH` and `MIINFER_SESSION_DIR` into `miinfer serve`.

---

## 4. Benchmark & Parity Qualification

Conducted with [`bench/v2_0028_persistent_session_bench.cpp`](../bench/v2_0028_persistent_session_bench.cpp):

```bash
./build/mi50-release/miinfer-v2-0028-persistent-session-bench \
  --model /home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf \
  --prefix 4096 --suffix 512
```

### 4.1 Measurement Results

| Execution Phase | Cold Baseline | Persistent Session Restore | Delta / Speedup |
|:---|---:|---:|---:|
| Prefix Ingestion ($P=4096$) | 19,495.97 ms | 491.17 ms (Disk $\rightarrow$ VRAM) | **39.7x faster** |
| Disk Write Latency | — | 574.38 ms (407.52 MiB @ 0.7 GB/s) | One-time cost |
| Full Prompt TTFT ($P=4608$) | 22,112.05 ms | 3,111.87 ms (Restore + Suffix) | **7.1x faster** |
| Suffix-Only TTFT ($S=512$) | N/A | 2,620.71 ms | ~2.6 s qualified |
| Generation Tokens (32 tok) | `[28130, 38208, 138784, ...]` | `[28130, 38208, 138784, ...]` | **100% IDENTICAL** |
| Token Parity Verification | — | — | **PASSED** |

---

## 5. Architectural Interpretation

1. **Practical Economics**:
   - Re-running cold prefill on 4096 tokens costs **19.5 seconds** of MI50 compute.
   - Restoring the serialized state from NVMe takes **491 milliseconds**.
   - This delivers a **39.7x reduction** in prompt preparation latency across server restarts, worker redeployments, or session resumption.
2. **Serving Implications**:
   - When serving coding agents or long-context sessions, prefix states can be safely evicted from precious VRAM to disk, freeing up memory while guaranteeing sub-second restoration upon reactivation.
3. **Determinism Guaranteed**:
   - Both GDN recurrent float matrices and GQA FP16 KV pairs are bit-preserved, guaranteeing zero numerical drift.

---

## 6. Decision & Roadmap

- **Status**: **KEEP & QUALIFY**.
- **Roadmap Sequence**:
  - **V2-0028** (Current): Persistent Prefix Cache & Session Restore Qualification — **DONE**.
  - **V2-0029** (Next): MMQ Useful-Compute Efficiency & Instruction-Amplification Analysis.
