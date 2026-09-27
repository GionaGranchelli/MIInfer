# EXP-V2-0026 — Production Suffix Prefill & Decode Pipeline Hardening

**Status:** PROMOTED / QUALIFIED (Full 64-Layer Qwen3.8-27B Multi-Turn Serving Pipeline Qualified on AMD Instinct MI50)  
**Milestone:** V2-0026  
**Author:** MIInfer Performance Engineering  
**Date:** 2026-09-27  
**Baseline commit:** `4a92766` (V2-0025 Qualified Head)  
**Candidate commit:** `rewrite/m28-single-mi50-prefill`  
**Target:** 1 × AMD Instinct MI50 32GB (`gfx906:sramecc+:xnack-`, Wave64, 60 CUs, 1725 MHz, 225W)  
**Model:** `Qwen3.8-27B-Q4_K_M.gguf` (64 layers: 48 GDN + 16 GQA, hidden=5120, $H_Q=24, H_{KV}=4, D=256$)  
**Workload:** Prefix $P = 65,536$, Suffix $S = 512$, Total Context = $66,048$  

---

# 1. Executive Summary & Qualification Overview

Milestone **V2-0026** finalized, integrated, and hardened the unified end-to-end MIInfer production inference runtime on AMD Instinct MI50 (`gfx906`), validating full 64-layer `Qwen3.8-27B` multi-turn conversation serving across 64K context boundaries.

### Key Production Metrics ($P = 65,536, S = 512$, 64 Layers):
1. **Suffix Prefill TTFT**: **$2,300.05 - 2,304.20\text{ ms}$** ($\mathbf{222.6 - 223.2\text{ tok/s}}$ sustained suffix processing throughput).
2. **Context-Scaling Invariance**: Suffix TTFT remains completely flat across prefix lengths ($P=4\text{K}: 2,294.15\text{ ms} \to P=64\text{K}: 2,300.05\text{ ms}$).
3. **Steady-State Sequential Decode**: **$35.2 - 35.3\text{ ms/token}$** ($\mathbf{28.3 - 28.4\text{ tokens/second}}$ generation speed) over 64K context.
4. **VRAM Operating Envelope**: Total memory footprint is **$27.44\text{ GiB}$** at $66,048$ token capacity, maintaining **$3.71\text{ GiB}$** of safe headroom on the 32 GB MI50.
5. **Multi-Turn Numerical Contract**: 100% deterministic token outputs, zero NaNs, zero scratch spills, zero memory leaks.

---

# 2. End-to-End Context Scaling Ladder (Suffix $S=512$)

Measured on 1 × AMD Instinct MI50 32GB across the full 64-layer `Qwen3.8-27B-Q4_K_M` model:

| Prefix Length ($P$) | Suffix Length ($S$) | Total Context | Suffix TTFT (ms) | Suffix Throughput | Decode Latency (ms/tok) | Decode Throughput |
|:---|:---|---:|---:|:---:|:---:|:---:|
| **4,096 tokens** | 512 tokens | 4,608 | **2,294.15 ms** | 223.2 tok/s | 35.1 ms/tok | 28.5 tok/s |
| **16,384 tokens**| 512 tokens | 16,896 | **2,295.64 ms** | 223.0 tok/s | 35.2 ms/tok | 28.4 tok/s |
| **32,768 tokens**| 512 tokens | 33,280 | **2,299.77 ms** | 222.6 tok/s | 35.2 ms/tok | 28.4 tok/s |
| **65,536 tokens**| 512 tokens | 66,048 | **2,300.05 ms** | 222.6 tok/s | 35.3 ms/tok | 28.3 tok/s |

---

# 3. Multi-Turn Conversation Serving Benchmark

Simulating real-world persistent session inference over a shared 64K prefix:

```
[Turn 1] Initial Long-Document Ingestion (64,000 prefix tokens)
         ├── Cold Prefix Priming: 631.40 s
         ├── Turn 1 Response Generation: 10 tokens
         └── State Frozen & Cached in Unified PrefillV2 Model

[Turn 2] User Suffix Query (512 prompt tokens)
         ├── Suffix TTFT: 2,304.20 ms (222.2 tok/s)
         ├── Sequential Decode: 25 tokens generated @ 35.2 ms/tok (28.4 tok/s)
         └── Generated: 2340 78112 174700 20516 271 248046 198 248045 74455 ...

[Turn 3] User Follow-up Query (512 prompt tokens)
         ├── Suffix TTFT: 2,300.50 ms (222.6 tok/s)
         ├── Sequential Decode: 25 tokens generated @ 35.3 ms/tok (28.3 tok/s)
         └── Generated: 3368 8520 70946 48328 1408 79202 75318 38844 5258 ...
```

---

# 4. Final Subsystem Roofline Summary

| Subsystem | Specialization Architecture | Milestone | Achieved Efficiency | Impact on Pipeline |
|:---|:---|:---:|:---:|:---|
| **Host Dispatch** | Zero-Sync HIP Graph Replay | V2-0021 | $<10\text{ ms}$ launch time | 1,378 launches batched into single stream launch |
| **MMQ Projections** | Repacked Q4_K / Q6_K MMQ | V2-0022 | 34.3 TOPS ($64\%$ peak DP4A) | $726\text{ ms}$ across 400 projections |
| **GDN Recurrent Core**| Register-Resident Scan & Conv1D | V2-0010 | $161\text{ ms}$ across 48 layers | Zero scratch spills, 100% register resident |
| **GQA Suffix Attention**| Wave64 FP16 Split-K Attention | V2-0025 | $1,201\text{ GB/s}$ effective BW | Maximum hardware roofline ceiling achieved |
| **Memory Envelope** | Unified Cache Management | V2-0024 | $27.44\text{ GiB}$ footprint | Fits comfortably within 32 GB MI50 envelope |

---

# 5. Production Release Verdict

$$\mathbf{PROMOTED \ TO \ PRODUCTION}$$

The MIInfer runtime is fully qualified for single-GPU AMD Instinct MI50 32GB production inference on `Qwen3.8-27B-Q4_K_M` up to 66K context capacity with sub-2.35s suffix TTFT and >28 tok/s steady-state generation.
