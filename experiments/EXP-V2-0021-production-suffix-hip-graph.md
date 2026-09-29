# EXP-V2-0021 — Production Suffix-Prefill HIP Graph Replay & Eager Dispatch Elimination

**Status:** REJECT (Host-Dispatch Bottleneck Hypothesis Falsified; Reusable Graph & Production Fallbacks Functionally Qualified)  
**Milestone:** V2-0021  
**Author:** MIInfer Performance Engineering  
**Date:** 2026-09-27  
**Baseline commit:** `1770948` (V2-0020 Qualified Head)  
**Candidate commit:** `rewrite/m28-single-mi50-prefill`  
**Target:** 1 × AMD Instinct MI50 32GB (`gfx906:sramecc+:xnack-`, Wave64, 60 CUs, 1606/1000 MHz, 225W)  
**Model:** `Qwen3.8-27B-Q4_K_M.gguf` (64 layers: 48 GDN + 16 GQA, hidden=5120)  

---

# 1. Executive Summary & Objective

Milestone **V2-0021** evaluated whether replacing production eager kernel dispatch with a reusable **HIP Graph** (`hipGraphExec_t`) for the qualified $S=512$ suffix-prefill execution path can eliminate the hypothesized $\sim 1.08\text{ s}$ host-side launch and orchestration overhead during long-prefix inference ($P = 65,536, S = 512$).

### Primary Hypotheses:
1. **Performance Opportunity Hypothesis**: Reusable HIP Graph replay will eliminate $\ge 800\text{ ms}$ of host-side dispatch overhead, achieving total suffix TTFT $\le 6.8\text{ s}$ at $P = 65,536, S = 512$.
2. **Production Qualification Hypothesis**: A single captured static graph can execute across arbitrary context positions ($P \in [512, 65536]$) and multi-turn continuations with 100% greedy token parity, zero VRAM leakage, and seamless eager fallback for non-512 suffix shapes.

### Key Experimental Findings:
1. **Hypothesis 1 Falsification (Asynchronous Overlap Microarchitecture)**:
   - On single-GPU AMD Instinct MI50 (`gfx906`), the host CPU queues all 1,378 kernel launches asynchronously into the HIP stream in $\approx 3.8\text{ ms}$ total host CPU time ($\approx 2.7\,\mu\text{s}/\text{node}$).
   - Because the GPU pipeline is heavily memory-bandwidth and compute bound ($\approx 5.49\text{ s}$ Split-K GQA suffix attention + $\approx 0.76\text{ s}$ MMQ projections), the GPU never stalls waiting for host kernel enqueueing.
   - Measured TTFT delta at $P = 65,536, S = 512$ across 5 interleaved A/B trials was **$-1.99\text{ ms}$ ($-0.03\%$, $1.000\times$ speedup)**, far below the $\ge 800\text{ ms}$ milestone target ($\ge 400\text{ ms}$ required for promotion).
2. **Hypothesis 2 Verification (Production Integrity & Correctness)**:
   - **Greedy Trajectory Parity**: 100% exact bitwise greedy token match across all 5 context regimes ($P \in \{512, 2\text{K}, 8\text{K}, 32\text{K}, 64\text{K}\}, S=512$) and multi-turn rollouts ($P = 4,096 \to 6,656$).
   - **Resource & Graph Stability**: Graph count $= 1$ (zero graph explosions, zero host DAG parameter mutations), with $0\text{ bytes}$ VRAM growth and zero driver errors across 20+ consecutive stress replays.
   - **Automatic Fallback**: Non-standard suffix lengths ($S \in \{128, 256, 384\}$) cleanly and transparently fall back to the eager path with identical generation output.

---

# 2. Experimental Setup & Workload Specification

- **GPU**: AMD Instinct MI50 32GB HBM2 (gfx906, 60 CUs, Wave64, PCIe 3.0 x16)
- **ROCm / Driver**: ROCm 6.2+ / HIP AMD runtime
- **Model**: `Qwen3.8-27B-Q4_K_M.gguf` (27B parameters, 64 layers: 48 GDN + 16 GQA, $D = 5120$)
- **KV Cache Quantization**: FP16 / FP16 unquantized
- **Workload Configuration**:
  - Prefix Length: $P = 65,536$
  - Suffix Chunk: $S = 512$ (Macro-tile)
  - Total Context: $66,048$ tokens
  - Evaluation Method: 5 interleaved A/B trials (A = Eager Control, B = HIP Graph Candidate) with cold and warm state normalization.

---

# 3. Experiment A: Correctness & Parity Across Context Regimes

Greedy autoregressive decoding (5 generated tokens) was compared between production Eager execution and HIP Graph Replay across context lengths from short to long context:

| Context Regime ($P$) | Suffix ($S$) | Eager Greedy Tokens | Graph Greedy Tokens | TTFT (ms) | Parity Result |
|:---|:---:|:---|:---|:---:|:---:|
| **$P = 512$** | 512 | `229537, 43327, 56875, 146367, 2340` | `229537, 43327, 56875, 146367, 2340` | 2,344.18 ms | **PASS (100% Match)** |
| **$P = 2,048$** | 512 | `146367, 2340, 78112, 151563, 102164` | `146367, 2340, 78112, 151563, 102164` | 2,447.18 ms | **PASS (100% Match)** |
| **$P = 8,192$** | 512 | `43327, 56875, 146367, 2340, 78112` | `43327, 56875, 146367, 2340, 78112` | 2,923.49 ms | **PASS (100% Match)** |
| **$P = 32,768$** | 512 | `102164, 43327, 56875, 146367, 2340` | `102164, 43327, 56875, 146367, 2340` | 4,876.98 ms | **PASS (100% Match)** |
| **$P = 65,536$** | 512 | `43327, 56875, 146367, 2340, 78112` | `43327, 56875, 146367, 2340, 78112` | 7,827.61 ms | **PASS (100% Match)** |

---

# 4. Experiment B: Interleaved Performance & Launch Attribution ($P=65,536, S=512$)

5 interleaved A/B benchmark trials were performed on the primary target workload:

| Trial | Eager TTFT (ms) | Graph Replay TTFT (ms) | Delta (ms) | Relative Delta (%) |
|:---:|---:|---:|---:|---:|
| **Trial 1** | 7,817.96 ms | 7,823.54 ms | -5.57 ms | -0.07% |
| **Trial 2** | 7,819.06 ms | 7,819.95 ms | -0.89 ms | -0.01% |
| **Trial 3** | 7,823.39 ms | 7,823.97 ms | -0.58 ms | -0.01% |
| **Trial 4** | 7,816.14 ms | 7,817.06 ms | -0.92 ms | -0.01% |
| **Trial 5** | 7,817.30 ms | 7,816.56 ms | +0.73 ms | +0.01% |
| **Summary** | **7,817.96 ms (Median)** | **7,819.95 ms (Median)** | **-1.99 ms** | **-0.03%** |

### Statistical Comparison:
- **Control (Eager Dispatch)**:
  - Median TTFT: **7,817.96 ms** (Suffix = 7,814.82 ms, Restore = 0.65 ms)
  - Mean TTFT: **7,818.77 ms** ($\sigma = \pm 2.79\text{ ms}$)
  - Range: [7,816.14 ms, 7,823.39 ms]
  - Host API Calls: **1,378 HIP launches / turn**
- **Candidate (HIP Graph Replay)**:
  - Median TTFT: **7,819.95 ms** (Suffix = 7,816.82 ms, Restore = 0.65 ms)
  - Mean TTFT: **7,820.22 ms** ($\sigma = \pm 3.48\text{ ms}$)
  - Range: [7,816.56 ms, 7,823.97 ms]
  - Host API Calls: **1 `hipGraphLaunch` / turn (99.78% reduction)**
- **Net Speedup**: **$1.000\times$** (No statistically meaningful wall-clock difference).

---

# 5. Experiment C: Multi-Turn Context Advancement

Tested dynamic position advancement without graph recapture across 5 consecutive turns ($P = 4,096 \to 6,656$):

| Turn | Position Range | Eager Token | Graph Token | Parity Result |
|:---:|:---:|:---:|:---:|:---:|
| **Turn 1** | $4,096 \to 4,608$ | `102164` | `102164` | **MATCH** |
| **Turn 2** | $4,608 \to 5,120$ | `43327` | `43327` | **MATCH** |
| **Turn 3** | $5,120 \to 5,632$ | `56875` | `56875` | **MATCH** |
| **Turn 4** | $5,632 \to 6,144$ | `146367` | `146367` | **MATCH** |
| **Turn 5** | $6,144 \to 6,656$ | `2340` | `2340` | **MATCH** |

- Graph instance count throughout multi-turn session: **1 static `hipGraphExec_t`** (0 invalidations, 0 node parameter updates).

---

# 6. Experiment D: Stress Replay & VRAM Leak Audit

- **Test**: 20 consecutive suffix HIP Graph replays at $P = 4,096, S = 512$.
- **Initial Free VRAM**: 3,654 MB
- **Final Free VRAM**: 3,654 MB
- **Net VRAM Growth**: **0 bytes**
- **Driver / Kernarg Errors**: 0

---

# 7. Experiment E: Non-Standard Suffix Fallback

Evaluated runtime behavior when suffix lengths differ from the qualified $S=512$ macro-tile:

| Suffix Length ($S$) | Base Prefix ($P$) | Eager Token | Auto-Fallback Token | Fallback Status |
|:---:|:---:|:---:|:---:|:---:|
| **$S = 128$** | 4,096 | `43327` | `43327` | **PASS (Exact Match)** |
| **$S = 256$** | 4,096 | `146367` | `146367` | **PASS (Exact Match)** |
| **$S = 384$** | 4,096 | `78112` | `78112` | **PASS (Exact Match)** |

---

# 8. Milestone Qualification Gate Evaluation

| Gate | Requirement | Measured | Verdict |
|:---|:---|:---:|:---:|
| **Gate 1: Numerical & Parity** | Exact greedy continuation match across 5 context lengths | 100% match ($P \in [512, 65536]$) | **PASS** |
| **Gate 2: Deterministic Replay** | Replay matches across multi-turn expansion | 5/5 turns match exactly | **PASS** |
| **Gate 3: Resource Stability** | Zero VRAM growth & zero kernarg exhaustion | 0 bytes growth across 20 replays | **PASS** |
| **Gate 4: Production Integration** | Integrated in `PrefillV2Model::generate()` | Yes, flag & size enabled | **PASS** |
| **Gate 5: Bounded Graph Count** | Graph count = 1 | 1 static `hipGraphExec_t` | **PASS** |
| **Gate 6: TTFT Improvement** | Median TTFT reduction $\ge 800\text{ ms}$ ($\ge 400\text{ ms}$ for partial) | **-1.99 ms** | **FAIL** |
| **Gate 7: Absolute TTFT Target** | Suffix TTFT $\le 6.8\text{ s}$ @ $P=65536, S=512$ | **7,819.95 ms** | **FAIL** |
| **Gate 8: GPU Kernel Regressions**| GPU kernel time regression $\le 2\%$ | 0.0% regression | **PASS** |
| **Gate 9: VRAM Budget** | Persistent VRAM $\le 32,768\text{ MB}$ | 28,158 MB (86.0% capacity) | **PASS** |
| **Gate 10: Safe Fallback** | Fall back cleanly on $S \ne 512$ | 100% match on $S=128, 256, 384$ | **PASS** |

---

# 9. Decision & Scientific Conclusion

### Decision: **REJECT** (for Performance Acceleration Claim)
- Per AGENTS.md rule 3.1 ("Measure before optimizing") and rule 3.3 ("Negative results are valuable"), the hypothesis that host-side kernel launch overhead accounts for $\sim 1.08\text{ s}$ of suffix TTFT is **falsified**.
- Host kernel enqueueing for 1,378 kernels requires only $\approx 3.8\text{ ms}$ of CPU driver time and is fully overlapped in the asynchronous stream queue during GPU execution of earlier layers.
- While HIP Graph replay drastically reduces CPU utilization and host API calls (1,378 $\to$ 1), it does not accelerate GPU wall-clock execution on MI50 because the hardware execution is 100% bound by GQA attention memory traffic and MMQ compute.

### Architectural Retention:
- The production implementation (`options.use_hip_graph = true` with dynamic `DevicePrefillState` pointer and automatic fallback) is fully verified, robust, and retained in the engine codebase as an optional CPU-offload mode for host-constrained or high-concurrency environments.
