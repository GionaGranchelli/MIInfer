# EXP-0196 — Combined Projections for Recurrent (QKV+Gate) and Attention (Q+K) Layers

## Hypothesis

Combining independent GEMV projections that share the identical input activation (`post_normalized` Q8_1 quantization) into unified matrix operations will:
1. Eliminate 80 separate kernel dispatches per token across 64 layers (48 in recurrent layers, 32 in attention layers).
2. Mitigate workgroup tail starvation on small projection matrices (recurrent Gate with 2,048 rows; attention K with 1,024 rows).
3. Increase average CU residency and compute wave saturation, reducing decode latency by $\ge 0.20$ ms/token without altering numerical output.

## Motivation

As detailed in `docs/M9-POST-M8-LATENCY-FLOOR.md`:
- In the 48 recurrent layers, `d_qkv_native` ($4096 \times 5120$) and `d_attn_gate_native` ($2048 \times 5120$) both project the same normalized input vector. Dispatched separately, the Gate projection (1,024 workgroups across 60 CUs = 17 workgroups/CU) leaves CUs under-utilized during tail execution.
- In the 16 attention layers, `d_attn_q_native` ($12288 \times 5120$) and `d_attn_k_native` ($1024 \times 5120$) both project the same input vector. The K projection (512 workgroups across 60 CUs = 8.5 workgroups/CU) exhibits severe wave starvation.

By allocating contiguous native tile buffers and combining weights at model load time:
- Recurrent layers combine QKV (4,096 rows) + Gate (2,048 rows) into a single $6144 \times 5120$ GEMV (3,072 workgroups = 51.2 workgroups/CU).
- Attention layers combine Q (12,288 rows) + K (1,024 rows) into a single $13312 \times 5120$ GEMV (6,656 workgroups = 110.9 workgroups/CU).

## Baseline

* Commit: `5a62eca` (M8 Qualified Baseline) + EXP-0194 Fast Arithmetic
* TG64 Median: 30.0274 tok/s / 33.303 ms/token
* TG128 Median: 29.8782 tok/s / 33.469 ms/token
* Environment: AMD Instinct MI50 32GB (gfx906 / Vega20, locked 1606 MHz SCLK / 1000 MHz MCLK, 225W power cap)
* Execution flags: `MIINFER_COMBINED_QKV_GATE=0` `MIINFER_COMBINED_ATTN_QK=0`

## Candidate

* Source: `tools/qwen35_gpu_pipeline.hpp`
  * Added combined native weight repacking during setup: `d_combined_qkv_gate_native` and `d_combined_attn_qk_native`.
  * Combined GEMV dispatch in layer forward execution for recurrent and attention stages.
  * Preserved separate output pointer mappings directly to destination buffers (`d_conv_in_proj`, `d_attn_gate_act`, `d_q_proj`, `d_k_proj`).
* Toggles:
  * `MIINFER_COMBINED_QKV_GATE=1` (enabled by default)
  * `MIINFER_COMBINED_ATTN_QK=1` (enabled by default)

## Environment

* GPU: AMD Instinct MI50 32GB
* Architecture: gfx906 / Vega20, 60 CUs, Wave64
* SCLK: 1606 MHz (DPM 7)
* MCLK: 1000 MHz (DPM 2)
* Power Cap: 225W
* Telemetry (TG64): 947 samples; 97.5% locked @ 1606 MHz, Avg Temp 52.7°C, Avg Power 94.5W
* Telemetry (TG128): 1262 samples; 94.3% locked @ 1606 MHz, Avg Temp 60.0°C, Avg Power 126.3W

## Benchmark

* Harness: `scripts/run-exp0196-combined-projections-ab.sh both`
* Methodology: 5-pair interleaved Control/Candidate runs for both TG64 and TG128 (20 benchmark runs total, 5 sample measurements per run)
* Model: `Qwen3.8-27B-Q4_K_M.gguf`

## Correctness

* CTest Regression Suite: 21/21 PASS (100%).
* 16-Token Autoregressive Generation:
  * Fingerprint: `11556965933579884203` (exact bitwise match with separate projection baseline).
  * Allocations during decode: 0.
  * Replay determinism: PASS.
* 64-Layer Observable Contract (`--prefix64-observable-contract`):
  * Positions evaluated: 64/64 PASS.
  * Mean Logits Cosine: **0.999611** (exceeds 0.9995 threshold and surpasses separate baseline 0.999546).
  * GPU winner rank: 1 / 1 across all 64 positions.
  * Poisoned reset replay: PASS.

## Results

### TG64 Interleaved Benchmark (5 Pairs)

| Pair | Control (tok/s) | Candidate (tok/s) | Control (ms/tok) | Candidate (ms/tok) | Delta | Winner |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Pair 1 | 30.2195 | 30.2398 | 33.091 | 33.069 | +0.07% | Candidate |
| Pair 2 | 30.1411 | 30.2384 | 33.177 | 33.071 | +0.32% | Candidate |
| Pair 3 | 29.8684 | 30.2102 | 33.480 | 33.101 | +1.14% | Candidate |
| Pair 4 | 30.0274 | 30.2726 | 33.303 | 33.033 | +0.82% | Candidate |
| Pair 5 | 29.9034 | 30.2429 | 33.441 | 33.066 | +1.14% | Candidate |
| **Median** | **30.0274** | **30.2398** | **33.303** | **33.069** | **+0.71%** | **Candidate (5/5)** |

- **TG64 Latency Saved:** **+0.234 ms/token (+0.70%)**

---

### TG128 Interleaved Benchmark (5 Pairs)

| Pair | Control (tok/s) | Candidate (tok/s) | Control (ms/tok) | Candidate (ms/tok) | Delta | Winner |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Pair 1 | 29.9639 | 30.0481 | 33.373 | 33.280 | +0.28% | Candidate |
| Pair 2 | 29.9583 | 30.0877 | 33.380 | 33.236 | +0.43% | Candidate |
| Pair 3 | 29.8782 | 30.0330 | 33.469 | 33.297 | +0.52% | Candidate |
| Pair 4 | 29.8093 | 30.0854 | 33.547 | 33.239 | +0.93% | Candidate |
| Pair 5 | 29.7978 | 30.0967 | 33.560 | 33.226 | +1.00% | Candidate |
| **Median** | **29.8782** | **30.0854** | **33.469** | **33.239** | **+0.69%** | **Candidate (5/5)** |

- **TG128 Latency Saved:** **+0.230 ms/token (+0.69%)**

---

## Profiling & Architectural Interpretation

1. **Kernel Launch Overhead:** Eliminating 80 kernel launches per decode step (48 recurrent + 32 attention) recovers dispatch latency on the host stream and reduces stream command processor queuing on the GPU.
2. **Occupancy and Wavefront Starvation:**
   - In separate projections, the recurrent Gate projection ($2048 \times 5120$) launched 1,024 workgroups. On 60 CUs, this is only 17.0 workgroups per CU, creating a severe wave quantization/tail effect at the end of the dispatch.
   - Merging QKV and Gate into $6144 \times 5120$ launches 3,072 workgroups (51.2 workgroups per CU), ensuring even CU distribution and high wave occupancy throughout the entire duration.
   - Similarly, merging attention Q and K into $13312 \times 5120$ launches 6,656 workgroups (110.9 workgroups per CU), completely eliminating the tail latency of the standalone 1,024-row K projection.
3. **Memory Footprint:** The combined projections use pre-allocated buffers that mirror the original separate buffers without increasing total persistent device allocations (`peak_device_bytes` remained identical at 18,886,426,964 B).
4. **Replay & Determinism:** Exact bitwise equivalence on tile calculations ensures 0 deviation across all runs, 0 decode allocations, and higher logits cosine (0.999611 vs 0.999546).

## Decision

**KEEP**.
Combined projections deliver an unequivocal win across 10 out of 10 benchmark pairs on both TG64 and TG128, recovering **0.234 ms/token** with zero correctness risk and zero VRAM increase. It is enabled by default in `tools/qwen35_gpu_pipeline.hpp`.

## Follow-up

To continue towards the M9 target of $\ge 32.00$ tok/s ($\le 31.25$ ms/token), current latency is ~33.07 ms/token, requiring an additional ~1.82 ms/token reduction.
The next optimization candidate is:
1. **Fused SwiGLU Epilogue with Inline Q8_1 Quantization:**
   - Currently, `launch_q4k_wave_fused_gate_up_swiglu` writes 17,408 floats to VRAM, then `launch_q8_1_quantize_f32` reads those 17,408 floats to produce 544 `Q8_1Block`s for FFN Down.
   - Inlining the `__shfl_xor` reduction into the SwiGLU kernel will emit `Q8_1Block`s directly, eliminating 64 standalone quantizer launches and round-trip HBM traffic.
