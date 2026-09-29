# EXP-V2-0027R — Cold-Prefill Arithmetic & Roofline Reconciliation

## 1. Executive Summary & Problem Statement

This document formally re-evaluates the mathematical performance model of MIInfer's cold prefill on AMD Instinct MI50 (`gfx906`, 60 CUs @ 1606 MHz) for `Qwen3.8-27B-Q4_K_M`.

A critical audit of the V2-0027 report revealed an arithmetic contradiction:
- Stated model work: **~48–54 GOP/token**
- Reported headline MMQ metric: **34.3 TOPS**
- Measured P512 throughput: **223.1 tok/s**

**The Arithmetic Inconsistency**:
If useful model compute were 34.3 TOPS at 54 GOP/token, throughput would be $\frac{34.3\text{ TOPS}}{54\text{ GOP/tok}} = \mathbf{635\text{ tok/s}}$, not 223 tok/s.
Conversely, at 223.1 tok/s end-to-end (249 tok/s for projections alone), the actual useful compute throughput achieved by the GPU is **11.87–12.0 TOPS**.

The previous conclusion that *"cold prefill is hardware-roofline closed at 69.5% efficiency"* conflated an isolated microbenchmark issue rate with useful end-to-end model arithmetic. 

**Reconciled Finding**:
MIInfer is currently achieving **11.87 TOPS of useful model compute during cold prefill**, which represents **24.1% of MI50's 49.33 TOPS peak DP4A capability**. Cold prefill is therefore **NOT hardware-roofline closed**. There is substantial theoretical compute headroom (~4×) to be investigated in future work.

---

## 2. Rigorous Model Arithmetic Derivation ($M \times N \times K$)

### 2.1 Layer-by-Layer Useful Matrix Multiply Work (Qwen3.8-27B)
*Counting standard Multiply-Accumulate operations: $1\text{ MAC} = 2\text{ FLOPs/OPs}$.*

#### A. Recurrent GDN Layer (48 Layers)
Per single token ($M=1$):
- **QKV Projection** [$5120 \times 10240$]: $2 \times 5120 \times 10240 = 104,857,600\text{ OPs}$ ($104.86\text{ MOPs}$)
- **Gate Projection** [$5120 \times 6144$]: $2 \times 5120 \times 6144 = 62,914,560\text{ OPs}$ ($62.91\text{ MOPs}$)
- **SSM Out Projection** [$6144 \times 5120$]: $2 \times 6144 \times 5120 = 62,914,560\text{ OPs}$ ($62.91\text{ MOPs}$)
- **FFN Gate Projection** [$5120 \times 17408$]: $2 \times 5120 \times 17408 = 178,257,920\text{ OPs}$ ($178.26\text{ MOPs}$)
- **FFN Up Projection** [$5120 \times 17408$]: $2 \times 5120 \times 17408 = 178,257,920\text{ OPs}$ ($178.26\text{ MOPs}$)
- **FFN Down Projection** [$17408 \times 5120$]: $2 \times 17408 \times 5120 = 178,257,920\text{ OPs}$ ($178.26\text{ MOPs}$)
- **Total per GDN Layer**: **$765.46\text{ MOPs/token}$**
- **48 GDN Layers Total**: $48 \times 765.46048\text{ MOPs} = \mathbf{36.742\text{ GOPs/token}}$

#### B. GQA Attention Layer (16 Layers)
Per single token ($M=1$):
- **Q Projection** [$5120 \times 6144$]: $2 \times 5120 \times 6144 = 62,914,560\text{ OPs}$
- **K Projection** [$5120 \times 1024$]: $2 \times 5120 \times 1024 = 10,485,760\text{ OPs}$
- **V Projection** [$5120 \times 1024$]: $2 \times 5120 \times 1024 = 10,485,760\text{ OPs}$
- **O Projection** [$6144 \times 5120$]: $2 \times 6144 \times 5120 = 62,914,560\text{ OPs}$
- **FFN Gate Projection** [$5120 \times 17408$]: $178,257,920\text{ OPs}$
- **FFN Up Projection** [$5120 \times 17408$]: $178,257,920\text{ OPs}$
- **FFN Down Projection** [$17408 \times 5120$]: $178,257,920\text{ OPs}$
- **Total per GQA Layer**: **$681.57\text{ MOPs/token}$**
- **16 GQA Layers Total**: $16 \times 681.5744\text{ MOPs} = \mathbf{10.905\text{ GOPs/token}}$

#### C. Full 64-Layer Useful Projection Arithmetic
- **Useful Projection Work per Token**: $36.742 + 10.905 = \mathbf{47.647\text{ GOPs/token}}$
- **Useful Non-Projection Work per Token** (Scan, Norms, RoPE, Softmax, LM Head): $\approx \mathbf{1.01\text{ GOPs/token}}$
- **Total Useful Work per Token**: $\mathbf{48.65\text{ GOPs/token}}$

---

## 3. Authoritative Reconciled Performance Metrics ($M=512$ Macro-Tile)

For a single MacroTile of $N = 512$ tokens:
- **Total Useful Projection Work**: $512 \times 47.647 \times 10^9 = \mathbf{24.395\text{ TOPs}}$ ($24.40\text{ TFLOPs}$)
- **Total Measured GPU Projection Time**: $\mathbf{2055.22\text{ ms}} = \mathbf{2.05522\text{ s}}$
- **Total Measured GPU Model Time (512t)**: $\mathbf{2294.84\text{ ms}} = \mathbf{2.29484\text{ s}}$

### Reconciled Metrics:
| Metric | Formula | Reconciled Value |
| :--- | :--- | :--- |
| **Useful MMQ Compute Throughput** | $\frac{24.395\text{ TOPs}}{2.05522\text{ s}}$ | **11.87 TOPS** |
| **End-to-End Compute Throughput** | $\frac{512 \times 48.65\text{ GOP}}{2.29484\text{ s}}$ | **10.85 TOPS** |
| **MI50 Peak DP4A Capability** | $60 \times 4 \times 16 \times 8 \times 1.606\text{ GHz}$ | **49.33 TOPS** |
| **Useful DP4A Hardware Efficiency** | $\frac{11.87\text{ TOPS}}{49.33\text{ TOPS}}$ | **24.06%** |
| **Pure MMQ Projection Throughput** | $\frac{512\text{ tokens}}{2.05522\text{ s}}$ | **249.1 tok/s** |
| **End-to-End Cold Prefill Throughput** | $\frac{512\text{ tokens}}{2.29484\text{ s}}$ | **223.1 tok/s** |

---

## 4. Context Scaling: P512 vs. Long-Context Cold Ingestion

In hybrid architectures (48 GDN recurrent + 16 GQA attention), cold prefill is not uniform across context lengths:

| Sequence Length | Tiling Structure | Wall Time (ms) | Cold Throughput (tok/s) | Marginal Cost (ms/tok) | Dominant Regimes |
| :--- | :--- | ---:| ---:| ---:| :--- |
| **P64** | 1 x 64 | 611.10 ms | 104.7 tok/s | 9.55 ms | CU occupancy / launch bound |
| **P128** | 1 x 128 | 714.53 ms | 179.1 tok/s | 5.58 ms | Tile ramp |
| **P512** | 1 x 512 | 2294.84 ms | 223.1 tok/s | 4.48 ms | MMQ compute bound (89.6% proj) |
| **P1024** | 2 x 512 | 4624.52 ms | 221.4 tok/s | 4.52 ms | Steady MMQ chunking |
| **P2048** | 4 x 512 | 9406.58 ms | 217.7 tok/s | 4.59 ms | Slight attention overhead |
| **P4096** | 8 x 512 | 19452.19 ms | 210.6 tok/s | 4.75 ms | Attention KV accumulation |
| **P8192** | 16 x 512 | 41383.86 ms | 198.0 tok/s | 5.05 ms | Attention quadratic cost visible |

At short to medium context ($P \le 2048$), cold prefill is strictly MMQ-dominated.
At long context ($P \ge 8192$), the 16 full-attention layers experience growing $O(N^2)$ KV-cache scanning costs, causing throughput to drop from 223 tok/s to 198 tok/s.

---

## 5. Where the Missing ~76% Compute Headroom Lies

Achieving 11.87 TOPS on a 49.33 TOPS chip means there is a **~4.15× gap** between current performance and theoretical peak DP4A.

The key factors accounting for this gap include:
1. **Dequantization & Non-DP4A Math**: Every block execution requires FP32 conversions, scaling products ($d \cdot d_x$), and minimum subtractions ($d_{min} \cdot minimum \cdot sum$), which execute on standard VALU pipelines and compete for instruction issue slots.
2. **LDS Synchronization & Staging Overhead**: The current kernel synchronizes across waves using LDS barriers per K-slice.
3. **MacroTile Geometry ($M=128, N=64$)**: Tile dimensions may not yet perfectly saturate all 60 CUs across uneven projection shapes (e.g. $K=17408$ vs $K=5120$).
4. **Vector Load Latency & Memory Divergence**: Memory bandwidth during activation gathering and weight staging.

---

## 6. Scientific Verdict & Updated Roadmap

### Project Record Disposition:
> **P512 MMQ kernel instruction throughput is provisionally qualified at 11.87 TOPS (223 tok/s end-to-end). Cold-prefill optimization is NOT hardware-roofline closed; significant theoretical headroom (~4×) remains in the DP4A SIMD path for future dedicated optimization.**

### Strategic Roadmap:
1. **Immediate Product Priority (V2-0028)**: Proceed with **V2-0028 — Persistent Prefix Cache & Session Restore Qualification**. In production serving, prompt state persistence bypasses cold prefill entirely on subsequent turns, achieving ~2.3s TTFT.
2. **Future Research Track (V2-0029+)**: Re-examine batched MMQ microarchitectures (LDS-free direct registers, fused FP32 scale pipelines, specialized shape tiles) with the goal of lifting useful compute throughput from 12 TOPS toward 25–35 TOPS.
