# EXP-V2-0014 — Prefix & State Reuse with Suffix-Only Prefill

## 1. Hypothesis
By introducing an explicit `ReusableContext` abstraction that preserves resident GQA KV cache entries ($0 \dots P-1$) in-place across the 16 full-attention layers and captures a compact contiguous GPU checkpoint ($151.50\text{ MiB}$) of the 48 GDN SSM recurrent and Conv1D history states at the prefix boundary, MIInfer can eliminate redundant prefix recomputation in multi-turn execution. For a $64\text{K}$ prefix with $512$ suffix tokens ($P=65536, S=512$), Turn 2 prefill TTFT will drop to $\le 15\%$ of cold full-prompt TTFT (achieving $> 6.7\times$ speedup) with zero decode degradation and bit-exact greedy token trajectory parity.

---

## 2. Architecture & Implementation
1. **Explicit ReusableContext Abstraction** (`include/miinfer/prefill_v2/reusable_context.hpp`, `src/prefill_v2/reusable_context.cpp`):
   - **`PrefixFingerprint`**: Model ID, quantization type, 64-bit FNV-1a token sequence hash, prefix length, and layout version (`kStateLayoutVersion = 20260925`).
   - **`GdnCheckpointStorage`**: Contiguous GPU allocation of $151.50\text{ MiB}$ ($158,859,264\text{ bytes}$) holding the exact recurrent state tensors ($48 \times 48 \times 128 \times 128 \times 4\text{B} = 144.0\text{ MiB}$) and Conv1D history buffers ($48 \times 4 \times 10240 \times 4\text{B} = 7.5\text{ MiB}$) for all 48 GDN layers at position $P$.
   - **Zero-Copy GQA KV Preservation**: Preserves Key and Value cache entries ($0 \dots P-1$) in-place in device memory with $0\text{ bytes}$ duplication and $0\text{ }\mu\text{s}$ transfer overhead.
2. **Deterministic Match & Invalidation**:
   - Compares incoming prompt against cached prefix fingerprint and token sequence.
   - Any mismatch (model, quantization, token corruption, short prompt, empty cache) falls back safely to cold prefill.
3. **Suffix-Only Forward Dispatch**:
   - In Turn 2, restores 48 GDN layer states in $\approx 0.67\text{ ms}$.
   - Dispatches Macro-512 prefill tiles starting strictly at `base_position = P` for the $S$ suffix tokens only.

---

## 3. Environment & Hardware State
- **Device**: 1 × AMD Instinct MI50 32GB (`gfx906:sramecc+:xnack-`, Wave64, 60 CUs, 1606 MHz SCLK, 1000 MHz MCLK, 225W, ROCm 7.1)
- **Model**: `Qwen3.8-27B-Q4_K_M.gguf` (64 layers: 48 GDN SSM + 16 GQA, hidden=5120, vocab=248320)
- **Base Commit**: `735ee7d10a17629ec2bfb591a94a956fc97920f6`
- **Implementation Commit**: `227320e43d9bf1eb628bbf14073ae00109968434`
- **Benchmark Command**: `./build/mi50-release/miinfer-prefix-state-reuse-bench`

---

## 4. Experimental Results

### Unit Validation & Failure Mode Suite
- `[1]` Exact Prefix Match: **PASS (ExactMatch)**
- `[2]` Model ID Mismatch: **PASS (ModelMismatch)**
- `[3]` Quantization Mismatch: **PASS (QuantizationMismatch)**
- `[4]` Token Content Mismatch: **PASS (PrefixMismatch)**
- `[5]` Prompt Shorter than Prefix: **PASS (PromptShorterThanPrefix)**
- `[6]` Empty Cache Check: **PASS (EmptyCache)**

### Measured Matrix: Cold vs Reusable Turn 2 Execution

| Scenario | Prefix / Suffix | Cold TTFT | Reuse TTFT | TTFT Speedup | Suffix Dispatched | Restore Latency | Decode Latency | Greedy Parity |
|:---|:---:|---:|---:|---:|---:|---:|---:|:---:|
| **Scenario 1: 4K Control** | 4K + 512 | 24,045.69 ms (24.0s) | **3,027.63 ms (3.0s)** | **7.94×** | 512 tokens | 0.716 ms | 38.55 ms/tok (-0.1%) | **EXACT MATCH** |
| **Scenario 2: 32K Agent** | 32K + 512 | 367,601.00 ms (6.1m) | **9,500.20 ms (9.5s)** | **38.69×** | 512 tokens | 0.677 ms | 47.79 ms/tok (+1.8%) | **EXACT MATCH** |
| **Scenario 3: 64K History (PRIMARY)** | 64K + 512 | 1,282,069.65 ms (21.4m) | **17,406.08 ms (17.4s)** | **73.66×** | 512 tokens | 0.672 ms | 60.56 ms/tok (+0.0%) | **EXACT MATCH** |
| **Scenario 4: 64K Large Suffix** | 64K + 4096 | 1,371,000.30 ms (22.9m) | **148,623.16 ms (2.5m)** | **9.22×** | 4,096 tokens | 0.668 ms | 62.04 ms/tok (-0.0%) | **EXACT MATCH** |

---

## 5. Architectural & Memory Accounting

### 1. VRAM Breakdown with Prefix Caching ($64\text{K} + 512$ Context)
- Base Model Weights: $22.42\text{ GiB}$
- Active KV Cache (67K capacity @ 64 KiB/token): $4.09\text{ GiB}$
- GDN Persistent State (48 layers): $0.15\text{ GiB}$
- **GDN Checkpoint Buffer (`GdnCheckpointStorage`)**: **$0.148\text{ GiB}$ ($151.50\text{ MiB}$)**
- Shared Workspace: $0.30\text{ GiB}$
- Activation Buffers: $0.02\text{ GiB}$
- **Total Static Resident VRAM**: **$27.12\text{ GiB}$** ($3.95\text{ GiB}$ free headroom on 32GB MI50).

### 2. Suffix-Only Execution Proof
Runtime instrumentation confirms that for Scenario 3 ($65,536\text{ prefix} + 512\text{ suffix}$):
- `stats.prefix_tokens_reused = 65536`
- `stats.suffix_tokens_dispatched = 512`
- `stats.gqa_kv_reused_tokens = 65536`
- Exact 512 tokens dispatched through Macro-512 scheduler. No kernels dispatched for the 65,536 prefix tokens.

---

## 6. Success Gates Evaluation

| Gate | Target Requirement | Measured Result | Status |
|:---|:---|:---|:---:|
| **Gate A: Correctness & Parity** | Greedy output parity = PASS; NaN/Inf = 0 | Exact token trajectory match across all scenarios | **PASSED** |
| **Gate B: Suffix-Only Dispatch** | No full-prefix kernel replay | 512 tokens dispatched for 64K+512 scenario | **PASSED** |
| **Gate C: 64K TTFT Speedup** | Reuse TTFT $\le 15\%$ of Cold TTFT | **$1.36\%$** ($17.4\text{s} / 1282.0\text{s} \implies \mathbf{73.66\times\text{ speedup}}$) | **PASSED (Beats 10% Stretch)** |
| **Gate D: Decode Preservation** | Decode regression $\le 3\%$ | $60.56\text{ ms/tok}$ vs $60.55\text{ ms/tok}$ ($+0.0\%$) | **PASSED** |
| **Gate E: Memory Overhead** | Compact cached-state footprint | Fixed $151.50\text{ MiB}$ checkpoint buffer | **PASSED** |

---

## 7. Decision
**QUALIFIED & PROMOTED**

Prefix and State Reuse with suffix-only prefill is verified, qualified, and promoted into the MIInfer V2 runtime.
