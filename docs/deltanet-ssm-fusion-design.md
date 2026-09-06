# Architectural Design: DeltaNet Recurrent SSM Core Fusion

## 1. Context and Problem Statement

In the 64-layer hybrid trunk of Qwen3.8-27B, there are **48 recurrent layers** utilizing the gated-DeltaNet linear attention architecture.
For each recurrent layer, the current execution pipeline launches 3 separate GPU kernels per token:

1. `launch_qwen35_conv_silu_split`: Updates 4-tap 1D depthwise convolution state across 6144 channels, computes SiLU activation, and splits into Query (2048), Key (2048), and Value (2048) float buffers.
2. `launch_qwen35_dual_head_l2_normalize`: Normalizes Query and Key head vectors (16 heads x 128 elements = 2048 elements each) using L2 norm across head dimension 128.
3. `launch_qwen35_deltanet_state_update_transposed_row_waves`: Updates the recurrent state matrix ($S \in \mathbb{R}^{16 \times 128 \times 128}$) via rank-1 Delta update with decay and produces the recurrent output projection input ($16 \times 128 = 2048$ elements).

### Overhead Identified:
- **Global Memory Traffic:** The intermediate tensors `query`, `key`, `query_norm`, and `key_norm` ($4 \times 8192\text{ bytes} = 32.768\text{ KB}$ per layer) are written to and read from device HBM across 3 distinct kernel boundaries.
- **Kernel Launch & Synchronization:** 3 distinct kernel launches $\times 48\text{ layers} = 144$ kernel launches per decode token. At ~3.5 $\mu$s host launch overhead, this consumes ~0.5 ms/token purely in dispatch.
- **Global Barriers:** Two implicit global memory synchronizations between conv, head-norm, and state-update.

---

## 2. Proposed Architecture: Fused Recurrent Core

We propose fusing these three sequential operations into a single cooperative kernel:
`launch_qwen35_fused_recurrent_core`

### 2.1 Workgroup and Wave Mapping
The recurrent state is partitioned into 16 value heads ($k = 16$). Each head possesses an inner state matrix of shape $[128, 128]$ (or transposed $[128, 128]$).
- Grid dimensions: 16 workgroups (1 workgroup per head).
- Workgroup size: 256 threads = 4 Wave64s.

### 2.2 Execution Plan within a Workgroup

1. **Step 1: Convolution & SiLU (In-Register / LDS)**
   - Channels $c \in [head \times 128, (head + 1) \times 128)$ are assigned to the workgroup's 4 waves (32 channels per wave).
   - Each lane loads the 4 convolution weights from constant/L1 cache and the history state (3 floats) from persistent VRAM.
   - Computes depthwise convolution + SiLU for Query, Key, and Value components.
   - Total intermediate data for 1 head: 128 query floats, 128 key floats, 128 value floats. This easily fits into workgroup LDS (384 floats = 1536 bytes).

2. **Step 2: Head L2 Normalization via Wave DPP / Shuffle**
   - Head dimension is exactly 128 floats.
   - With 2 waves cooperating (or 1 wave per 64-element half):
     - Each lane squares its elements and performs a 64-lane DPP tree reduction (`__builtin_amdgcn_update_dpp`).
     - LDS exchange combines the two wave sums to produce $\sum x_i^2$.
     - Computes $rsqrt(\sum x_i^2 + \epsilon)$ in a single scalar register.
     - Multiplies elements by the normalizer in registers.
   - **Zero global memory traffic**: `query_norm` and `key_norm` remain strictly in LDS/VGPRs!

3. **Step 3: Transposed DeltaNet Recurrent State Update**
   - Workgroup reads the normalized Key ($k \in \mathbb{R}^{128}$), normalized Query ($q \in \mathbb{R}^{128}$), Value ($v \in \mathbb{R}^{128}$), scalar decay $\alpha$, and scalar rate $\beta$.
   - The 4 waves update the $128 \times 128$ persistent state matrix:
     $$S_{t}[i, j] = S_{t-1}[i, j] \cdot \alpha + k_i \cdot (\beta \cdot v_j - (S_{t-1}^T q)_j)$$
   - Writes only the final recurrent output vector ($128$ floats per head) and the updated state matrix to VRAM.

---

## 3. Amdahl Impact & Expected Savings

| Metric | Current 3-Kernel Chain | Fused Recurrent Core | Delta |
|---|---:|---:|---:|
| Kernel launches / layer | 3 | 1 | -2 (-96 per token) |
| Global VRAM writes / layer | Conv: 24 KB, Norm: 16 KB, State: 65 KB | State: 65 KB, Out: 0.5 KB | **-40 KB / layer (-1.92 MB/tok)** |
| Host dispatch latency | ~10.5 $\mu$s / layer | ~3.5 $\mu$s / layer | **-0.34 ms/token** |
| Measured GPU kernel execution | ~0.115 ms / layer | ~0.082 ms / layer | **-1.58 ms/token** |
| **Total projected saving** | **~5.52 ms/token** | **~3.94 ms/token** | **~1.58 - 1.92 ms/token (+2.5-3.0% TG)** |

---

## 4. Implementation & Validation Sequence

1. **Host Reference Prototype**: Validate fused arithmetic bit-exactness against `qwen35_conv_reference` + `qwen35_head_l2_normalize_reference` + `qwen35_deltanet_reference`.
2. **Correctness Fixture**: Verify against the 64-position teacher-forced reference (`/tmp/m6a273-reference-p12`). Max relative difference must remain $\le 10^{-5}$.
3. **A/B Benchmark Gate**: Single-variable qualification under manual DPM 1606/1000 MHz.
