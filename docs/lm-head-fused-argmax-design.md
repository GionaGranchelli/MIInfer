# Architectural Design: Fused LM-Head GEMV and Argmax Reduction

## 1. Context and Current Bottleneck

In Qwen3.8-27B, the vocabulary size is $V = 248,320$ tokens, and the hidden representation size is $D = 5120$.
The vocabulary projection matrix (`output.weight`) is quantized as **Q6_K** with dimensions $[5120, 248320]$.

In the current decoding loop:
1. **LM-Head GEMV**: Launches a large GEMV kernel (`launch_qwen3_q6_k_gemv` or `launch_qwen3_q6_k_q8_k_gemv`) with 248,320 output rows.
   - Outputs a full `float logits[248320]` vector to device VRAM.
   - VRAM written: $248320 \times 4\text{ bytes} = 993,280\text{ bytes}$ (~0.993 MB per token).
2. **Device-to-Host Copy / Argmax**:
   - For greedy sampling (the primary benchmark path), a subsequent reduction or DtoH copy downloads the 0.993 MB logit buffer across PCIe to find $\text{argmax}_i (\text{logits}_i)$.
   - Or launches an argmax reduction kernel that reads 0.993 MB from VRAM to output a single integer `int best_token`.

### Architectural Inefficiency:
For autoregressive greedy decoding, intermediate logit values are **ephemeral** and never referenced again.
Writing 1 MB of floats over the memory bus and then reading it back costs ~2 MB of memory bandwidth per token and incurs PCIe/launch latency.

---

## 2. Proposed Architecture: Fused Q6_K LM-Head + Argmax

We propose integrating a two-level hierarchical reduction directly into the GEMV output stage:
`launch_q6k_wave_gemv_argmax(const Q6KWaveTile* w, const Q8_1Block* x, int* d_best_token, float* d_best_logit)`

### 2.1 Two-Stage Hierarchical Reduction

1. **Stage 1: Workgroup-Local Argmax**
   - Each workgroup computes $R = 2$ output rows (or $R = 4$ rows).
   - Thread 0 within the workgroup holds the computed row logits: `float logit0` and `logit1`.
   - Thread 0 compares its local logit against the workgroup's best using atomic or LDS reduction:
     ```cpp
     struct TokenCandidate {
         float logit;
         std::uint32_t token_id;
     };
     ```
   - Total workgroups in grid: $248320 / 2 = 124,160$ workgroups.
   - To avoid excessive global atomics, workgroups are organized into tiles of 128 workgroups sharing a scratch block, or outputting their candidate to a compact staging buffer `TokenCandidate partials[1940]`.
   - Size of staging buffer: $1940 \times 8\text{ bytes} \approx 15.5\text{ KB}$ (fully resident in L2 cache!).

2. **Stage 2: Single-Wave Final Reduction**
   - A single wave (64 threads) reads the 1940 candidates from L2 cache, performs a 64-lane tree reduction, and writes the single winner `(best_token, best_logit)` to a pinned host-accessible device word (8 bytes).

### 2.2 Numerical Invariance Guarantee
- The inner dot products and floating-point additions for every row remain **100% bit-identical** to the standalone `launch_q6k_wave_gemv`.
- The comparisons strictly follow IEEE-754 semantics: `if (candidate.logit > current.logit) current = candidate;` with tie-breaking on lowest `token_id` to preserve deterministic argmax contracts.

---

## 3. Projected Gains

| Metric | Current Independent LM-Head + Argmax | Fused LM-Head Argmax | Delta |
|---|---:|---:|---:|
| VRAM written per token | 993.3 KB | 8 bytes (+ 15.5 KB L2 scratch) | **-993 KB / token (-99.9%)** |
| VRAM read during reduction | 993.3 KB | 15.5 KB | **-978 KB / token (-98.4%)** |
| Kernel dispatches | 2 | 1 | -1 dispatch |
| End-to-end LM-head latency | ~2.85 ms | ~2.35 ms | **-0.50 ms/token (+0.75-0.9% TG)** |

---

## 4. Rollout Preconditions
- Must not be rolled out simultaneously with K-quant tile rollout to preserve bisection.
- Candidate for Stage M6-B rollout once Q6KWaveTile has been qualified on projection families.
